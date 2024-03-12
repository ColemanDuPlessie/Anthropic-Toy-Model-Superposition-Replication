import math
import torch
from torch import nn
from matplotlib import pyplot as plt

torch.manual_seed(1)

DATA_SIZE = 80
MODEL_SIZE = 20
DATA_SPARSITY = 0.03
DATA_IMPORTANCE_DECAY = 0.9

LEARNING_RATE = 0.001
DECAY_PER_EPOCH = 0.99995

BATCH_SIZE = 32768
NUM_BATCHES = 8192

def gen_sparse_dummy_data(size, sparsity=0.5, num_batches=1):
    nonsparse_data = torch.rand((num_batches, size))
    mask = torch.rand((num_batches, size)) < sparsity
    return nonsparse_data*mask

def decaying_importance_MSE_loss(output, target, decay_rate=0.9):
    error = (output - target)**2
    importance = torch.Tensor([decay_rate**i for i in range(error.shape[-1])])
    return torch.mean(error*importance)

class AutoencoderToyModel(nn.Module):
    
    def __init__(self, in_out_size, hidden_size):
        super().__init__()
        self.in_out_size = in_out_size
        self.hidden_size = hidden_size
        self.w = nn.Parameter(torch.empty(in_out_size, hidden_size))
        self.b = nn.Parameter(torch.empty(in_out_size,))
        stdv = 1.0 / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        self.hidden_values = torch.matmul(x, self.w)
        return nn.functional.relu(torch.matmul(self.hidden_values, self.w.t())+self.b)
    
    def plot_weights(self):
        fig, axes = plt.subplots(ncols=2)
        
        min_value = min(torch.min(self.w).item(), torch.min(self.b).item())
        max_value = max(torch.max(self.w).item(), torch.max(self.b).item())

        ax1, ax2 = axes

        im1 = ax1.matshow(torch.matmul(self.w, self.w.t()).detach(), vmin=min_value, vmax=max_value)
        im2 = ax2.matshow(self.b.detach().unsqueeze(1), vmin=min_value, vmax=max_value)

        fig.colorbar(im1, ax=ax2)
        fig.show()
    
    def get_norm(self):
        return torch.norm(self.w, p='fro').item()
    
    def get_avg_dims_per_feature(self):
        return self.hidden_size/(self.get_norm()**2)
        

model = AutoencoderToyModel(DATA_SIZE, MODEL_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # TODO hyperparameter tuning?
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_PER_EPOCH)

losses = []
dims_per_feature = []

for i in range(NUM_BATCHES):
    data = gen_sparse_dummy_data(DATA_SIZE, DATA_SPARSITY, BATCH_SIZE)
    prediction = model(data)
    
    loss = decaying_importance_MSE_loss(prediction, data, DATA_IMPORTANCE_DECAY)
    losses.append(loss.item())
    dims_per_feature.append(model.get_avg_dims_per_feature())
    if i % 128 == 127 or i == 0:
        print(f"Batch {i} loss: {loss:.6f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

plt.plot(losses)
plt.yscale("log")
plt.show()

plt.plot(dims_per_feature)
plt.show()

model.plot_weights()