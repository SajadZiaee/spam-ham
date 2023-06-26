import torch
import torch.nn as nn
from torchviz import make_dot


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


# Define an instance of the network
input_dim = 3000  # Replace with the appropriate input dimension
net = Net(input_dim)

# Create a dummy input tensor for visualization
dummy_input = torch.randn(1, input_dim)

# Generate the graph and save it as PDF
graph = make_dot(net(dummy_input), params=dict(net.named_parameters()))
graph.render(filename='network_structure', format='pdf')
