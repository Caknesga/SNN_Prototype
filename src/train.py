import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen, functional as SF, Leaky

# ------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------
def main():
    # ------------------------------
    # 1. Parameters
    # ------------------------------
    batch_size = 128
    num_epochs = 2           # increase for better accuracy
    num_steps = 50           # simulation time steps
    beta = 0.9               # membrane decay constant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 2. Data
    # ------------------------------
    data_root = "data"  # folder containing MNIST/raw/*.gz
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])

    train_ds = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,     # set True only once if you need to download
        transform=transform
    )
    test_ds = datasets.MNIST(
        root=data_root,
        train=False,
        download=False,
        transform=transform
    )

    # safer on macOS — avoid multiprocessing here
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0)

    


    # ------------------------------
    # 3. Define the SNN
    # ------------------------------
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 256)
            self.lif1 = Leaky(beta=beta)
            self.fc2 = nn.Linear(256, 10)
            self.lif2 = Leaky(beta=beta)

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec = []

            for step in range(num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)

            return torch.stack(spk2_rec)

    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    # ------------------------------
    # 4. Training Loop
    # ------------------------------
    print("Starting training...\n")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for data, target in train_loader:
            data = data.view(batch_size, -1).to(device)
            target = target.to(device)
            data = spikegen.rate(data, num_steps=num_steps)

            spk_rec = net(data)
            spk_sum = spk_rec.sum(0)            # [B, 10]
            target_onehot = F.one_hot(target, 10).float()  # [B, 10]
            loss = F.mse_loss(spk_sum, target_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # ------------------------------
    # 5. Evaluation
    # ------------------------------
    print("\nEvaluating on test data...")
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(batch_size, -1).to(device)
            data = spikegen.rate(data, num_steps=num_steps)
            spk_rec = net(data)
            out = spk_rec.sum(0)
            pred = out.argmax(1)
            correct += (pred.cpu() == target).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test accuracy: {accuracy:.2f}%")

# ------------------------------
# 6. Safe entry point (important for macOS)
# ------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
