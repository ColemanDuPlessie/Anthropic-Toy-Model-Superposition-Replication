import torch
from torch import nn

torch.manual_seed(1)

DATA_SIZE = 10
MODEL_SIZE = 5
DATA_SPARSITY = 0.2
DATA_IMPORTANCE_DECAY = 0.75

LEARNING_RATE = 0.001

BATCH_SIZE = 128
NUM_BATCHES = 4096

def gen_sparse_dummy_data(size, sparsity=0.5, num_batches=1):
    nonsparse_data = torch.rand((num_batches, size))
    mask = torch.rand((num_batches, size)) < sparsity
    return nonsparse_data*mask

def decaying_importance_MSE_loss(output, target, decay_rate=0.9):
    error = (output - target)**2
    importance = torch.Tensor([decay_rate**i for i in range(error.shape[-1])])
    return torch.sum(error*importance)

class AutoencoderToyModel(nn.Module):
    
    def __init__(self, in_out_size, hidden_size):
        super().__init__()
        self.w = nn.Parameter(torch.randn((in_out_size, hidden_size)))
        self.b = nn.Parameter(torch.randn((in_out_size,)))
    
    def forward(self, x):
        self.hidden_values = torch.matmul(x, self.w)
        return nn.functional.relu(torch.matmul(self.hidden_values, self.w.t())+self.b)

model = AutoencoderToyModel(DATA_SIZE, MODEL_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # TODO hyperparameter tuning?

for i in range(NUM_BATCHES):
    data = gen_sparse_dummy_data(DATA_SIZE, DATA_SPARSITY, NUM_BATCHES)
    prediction = model(data)
    
    loss = decaying_importance_MSE_loss(prediction, data, DATA_IMPORTANCE_DECAY)
    if i % 128 == 127 or i == 0:
        print(f"Batch {i} loss: {loss:.5f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()