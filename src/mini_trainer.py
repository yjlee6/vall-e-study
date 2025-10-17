import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

'''
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
'''

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4):
        super().__init__() 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim    

        self.input_fc = nn.Linear(input_dim, hidden_dim) # 입력 차원을 임베딩 차원으로

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, # 입출력 벡터 차원 / 트랜스포머 내부 히든 크기
            nhead=num_heads, # 멀티헤드어텐션 헤드 수
            dim_feedforward=hidden_dim * 4, # feedforward 네트워크의 내부 차원 / *4로 차원 확장
            dropout=0.1, 
            activation='relu',
            batch_first=True, # 배치 차원이 첫 번째 차원인지 여부 (b, seq, feature)
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(hidden_dim, output_dim) # 임베딩 차원을 출력 차원으로

    def forward(self, x):
        """
        x: (batch_size, seq_length, input_dim)
        """
        x = self.input_fc(x)  # (batch_size, seq_length, input_dim) -> (batch_size, seq_length, hidden_dim)
        x = self.transformer(x)  # (batch_size, seq_length, hidden_dim)
        x = self.output_fc(x)  # (batch_size, seq_length, hiddien_dim) -> (batch_size, seq_length, output_dim)
        return x

def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, rank, writer=None):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        y = y.unsqueeze(1)  # (batch_size, 1, output_dim)

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
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            y = y.unsqueeze(1)  # (batch_size, 1, output_dim
            
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

    model = TransformerModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=2,
        num_heads=4 
    )

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