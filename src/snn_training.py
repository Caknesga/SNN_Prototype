import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import snntorch as snn
from snntorch.spikevision import spikedata

# ------------------------------
# 1. Parameters
# ------------------------------
batch_size = 64
num_epochs = 1           # increase for better results
beta = 0.9               # membrane decay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2. Dataset: Spiking MNIST
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

train_dataset = spikedata.SpikingMNIST(
    root="data/",
    train=True,
    transform=transform,
    download=True
)
test_dataset = spikedata.SpikingMNIST(
    root="data/",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_steps = train_dataset[0][0].shape[0]  # temporal length of spike trains

# ------------------------------
# 3. Define simple SNN
# ------------------------------
class SNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(100, 10)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step].view(x.size(1), -1))  # flatten per time step
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

net = SNNNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ------------------------------
# 4. Training loop
# ------------------------------
print("Training on STMNIST...\n")
for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    for i, (spike_data, targets) in enumerate(train_loader):
        spike_data = spike_data.to(device)
        targets = targets.to(device)

        spk_rec = net(spike_data)
        spk_sum = spk_rec.sum(dim=0)
        loss = loss_fn(spk_sum, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} complete - Loss: {total_loss / len(train_loader):.4f}")

# ------------------------------
# 5. Evaluation
# ------------------------------
print("\nTesting...\n")
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for spike_data, targets in test_loader:
        spike_data = spike_data.to(device)
        targets = targets.to(device)
        spk_rec = net(spike_data)
        spk_sum = spk_rec.sum(dim=0)
        preds = spk_sum.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
