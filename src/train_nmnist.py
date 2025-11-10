#essential libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import snntorch as snn
from snntorch import functional as SF # NOT REALLY NECCESARY
import numpy as np 

# ------------------------------
# 1. Parameters
# ------------------------------
batch_size = 64 #number of samples processed in one optimizer(backpropagation) step, 64-256 is fine on CPU, 
num_epochs = 1  #maximum 2-3 is fine
beta = 0.9      #LIF leak factor per simulation step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #run on GPU if present; else CPU

# ------------------------------
# 2. Load NMNIST Dataset via Tonic and transform
# ------------------------------
sensor_size = tonic.datasets.NMNIST.sensor_size #Instead of frames, it outputs spike events with: (x,y,polarity,timestamp)

frame_transform = transforms.Compose([ #defines how raw event streams are converted into tensor frames usable by PyTorch. 
    transforms.Denoise(filter_time=10000), #Removes isolated or spurious spikes that occur within 10 ms of inactivity.                       
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000)  #, adds a time dimension so our tensor will be 4D, ! smaller timewindow, smaller beta
])

#train and test dataset donwkload and transform 
train_dataset = tonic.datasets.NMNIST(
    save_to="data/", transform=frame_transform, train=True
)
test_dataset = tonic.datasets.NMNIST(
    save_to="data/", transform=frame_transform,train=False
)

#following method can be optimized
def pad_to_fixed_length(batch, max_time=300): #It converts irregular event sequences into fixed-length tensor for back-prop
    data_list, targets = zip(*batch)

    padded_data = []
    for data in data_list:
        if isinstance(data, np.ndarray):   # convert NumPy
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

#train and test data loader using Dataloader from torch, it is for back-prop neccesary
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, #Feeds batches of NMNIST samples to your network during backpropagation
                          num_workers=0, collate_fn=lambda b: pad_to_fixed_length(b, 300))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=lambda b: pad_to_fixed_length(b, 300)) #maybe we can decrease the training time with num_workers>0 ?

# ------------------------------
# 3. Define Simple SNN without convolution
# ------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 2 * 34 * 34   # DVS polarity channels × height × width
        self.fc1 = nn.Linear(in_features, 100)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(100, 10)
        self.lif2 = snn.Leaky(beta=beta)
        #2 layer is enoguh fot NMNIST dataset, Loihi 2 uses 3-5 layer
     
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
        spk_sum = spk_rec.sum(dim=1)
        preds = spk_sum.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Test Accuracy: {100.0 * correct / total:.2f}%")
