import math
import torch
from torch import nn
from matplotlib import pyplot as plt

torch.manual_seed(1)

DATA_SIZE = 6
CORRELATED_GROUP_SIZE = 2
MODEL_SIZE = 2
DATA_SPARSITIES = [0.01, 0.1, 0.2, 0.4, 0.8]

LEARNING_RATE = 0.001
DECAY_PER_EPOCH = 0.99995

BATCH_SIZE = 32768
NUM_BATCHES = 16384

GRAPHING_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

def gen_sparse_dummy_data(size, sparsity=0.5, num_batches=1):
    nonsparse_data = torch.rand((num_batches, size))
    mask = torch.rand((num_batches, size)) < sparsity
    return nonsparse_data*mask

def gen_sparse_correlated_dummy_data(size, sparsity=0.5, correlated_group_size=2, num_batches=1):
    assert size % correlated_group_size == 0 # The input channels must be evenly divided into correlated groups
    nonsparse_data = torch.rand((num_batches, size))
    mask = torch.kron(torch.rand((num_batches, size//correlated_group_size)), torch.ones(1, correlated_group_size)) < sparsity
    return nonsparse_data*mask

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
    
    def plot_2D_weights(self, group_size=1):
        assert self.hidden_size == 2
        for feature in range(self.in_out_size):
            color = GRAPHING_COLORS[(feature//group_size) % len(GRAPHING_COLORS)]
            weights = self.w[feature].detach()
            plt.plot(weights[0], weights[1], 'o', color=color)
            plt.plot((0, weights[0]), (0, weights[1]), color=color)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()


loss_func = nn.MSELoss()
past_models = []
for sparsity in DATA_SPARSITIES:
    print("\n"*5)
    print(f"BEGINNING TRIAL RUN {len(past_models)} WITH SPARSITY {sparsity}")
    print("\n"*5)
    
    model = AutoencoderToyModel(DATA_SIZE, MODEL_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # TODO hyperparameter tuning?
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_PER_EPOCH)
    
    losses = [] 
    for i in range(NUM_BATCHES):
        data = gen_sparse_correlated_dummy_data(DATA_SIZE, sparsity, CORRELATED_GROUP_SIZE, BATCH_SIZE)
        prediction = model(data)
        
        loss = loss_func(prediction, data)
        losses.append(loss.item())
        if i % 128 == 127 or i == 0:
            print(f"Batch {i} loss: {loss:.6f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    past_models.append(model)

    plt.plot(losses)
    plt.yscale("log")
    plt.show()
    
    model.plot_2D_weights(CORRELATED_GROUP_SIZE)