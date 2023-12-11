import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 2 Convolution Layers
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * 4 * 4, 128) 
        self.relu = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(128, 62)
        self.soft_max = nn.Softmax(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

#test 
