import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class MiniModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, rank, writer=None):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}") 
    avg_loss = total_loss / len(dataloader)
    print(f"Rank {rank}, Epoch {epoch}, Average Loss: {avg_loss}")

    if writer:
        print(f"wrtiting train loss to TensorBoard", {'Loss/train': avg_loss, 'epoch': epoch})
        writer.add_scalar('Loss/train', avg_loss, epoch)

    return avg_loss

def valid_one_epoch(model, dataloader, loss_fn, epoch, rank, writer=None):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Rank {rank}, Epoch {epoch}, Validation Average Loss: {avg_loss}")

    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, exp_dir):
    """ Saves model and optimizer states to a checkpoint file."""
    exp_dir = Path(exp_dir)
    ckpt_path = exp_dir / f"checkpoint_epoch_{epoch}.pth"

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, ckpt_path)

    print(f"Checkpoint saved at {ckpt_path}")

def load_checkpoint(model, optimizer, exp_dir): 
    ckpt_files = sorted(Path(exp_dir).glob("checkpoint_epoch_*.pth"))
    if not ckpt_files:
        print("No checkpoint found.")
        return 0
    
    latest_ckpt = ckpt_files[-1]
    checkpoint = torch.load(latest_ckpt)
    print("!!!!!! checkpoint items:", checkpoint.keys())
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Loaded checkpoint from {latest_ckpt}, starting from epoch {start_epoch}")
    return start_epoch


def run(rank, world_size, args):
    print(f"Running on rank {rank} out of {world_size} processes")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=exp_dir / "tensorboard")

    X = torch.randn(10000, 128)
    Y = torch.randn(10000, 128)

    dataset = TensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = MiniModel(args.input_dim, args.hidden_dim, args.output_dim)
    print("파라미터 개수:", sum(p.numel() for p in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    start_epoch = 1
    if args.resume:
        print("Resuming from checkpoint...")
        start_epoch = load_checkpoint(model, optimizer, args.exp_dir)
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, rank, writer)
        val_loss = valid_one_epoch(model, val_loader, loss_fn, epoch, rank, writer)

        save_checkpoint(model, optimizer, epoch, args.exp_dir)
        print(f"Rank {rank}, Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        print("-"*50)
    writer.close()
    print(f"TensorBoard logs saved to {exp_dir / 'tensorboard'}")

def main():
    parser = argparse.ArgumentParser() # Object for parsing command line strings into Python objects.
    # print(help(parser))
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes to launch")
    parser.add_argument("--exp_dir", type=str, default="./exp", help="Experiment directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--input_dim", type=int, default=128, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--output_dim", type=int, default=128, help="Output dimension")     
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the synthetic dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args() # store the parsed arguments in args
    print(f"Parsed arguments: {args}")

    world_size = args.world_size
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True) 
        # mp.spawn is used to launch multiple processes for distributed training 
        # nprocs specifies the number of processes to launch
        # join=True means the main process will wait for all spawned processes to finish
    else:
        run(0, world_size, args)

if __name__ == "__main__":
    main()