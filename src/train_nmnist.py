import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import snntorch as snn
from snntorch import functional as SF
import numpy as np

# ------------------------------
# 1. Parameters
# ------------------------------
batch_size = 64
num_epochs = 1
beta = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2. Load NMNIST Dataset via Tonic
# ------------------------------
sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
])


train_dataset = tonic.datasets.NMNIST(
    save_to="data/", transform=frame_transform, train=True
)
test_dataset = tonic.datasets.NMNIST(
    save_to="data/", transform=frame_transform,train=False
)

def pad_to_fixed_length(batch, max_time=300):
    data_list, targets = zip(*batch)

    padded_data = []
    for data in data_list:
        if isinstance(data, np.ndarray):   # convert NumPy → Tensor if needed
            data = torch.tensor(data, dtype=torch.float32)

        t, c, h, w = data.shape
        if t > max_time:
            data = data[:max_time]
        else:
            pad = torch.zeros((max_time - t, c, h, w))
            data = torch.cat((data, pad), dim=0)
        padded_data.append(data)

    data_batch = torch.stack(padded_data)
    targets = torch.tensor(targets, dtype=torch.long)
    return data_batch, targets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, collate_fn=lambda b: pad_to_fixed_length(b, 300))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=lambda b: pad_to_fixed_length(b, 300))

num_steps = train_dataset[0][0].shape[0]  # temporal length

# ------------------------------
# 3. Define Simple SNN
# ------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 2 * 34 * 34   # DVS polarity channels × height × width
        self.fc1 = nn.Linear(in_features, 100)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(100, 10)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for step in range(x.size(1)):   # iterate over time dimension
            cur1 = self.fc1(x[:, step].view(x.size(0), -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=1)

net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ------------------------------
# 4. Training
# ------------------------------
print("Training on NMNIST...\n")
for epoch in range(num_epochs):
    total_loss = 0
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        spk_rec = net(data)
        spk_sum = spk_rec.sum(dim=1)
        loss = loss_fn(spk_sum, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------
# 5. Evaluation
# ------------------------------
print("\nTesting...")
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = net(data)
        spk_sum = spk_rec.sum(dim=0)
        preds = spk_sum.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Test Accuracy: {100.0 * correct / total:.2f}%")
