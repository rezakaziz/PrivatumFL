import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 2 Convolution Layers
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(128, 64, 3)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * 5 * 5, 128) 
        self.relu = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(128, 62)
        #self.soft_max = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 5 * 5)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x 

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def load_data():
    trf = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])
    trainset = MNIST("./data",train=True,download=True,transform=trf)
    testset =  MNIST("./data",train=False,download=True,transform=trf)
    return DataLoader(trainset,batch_size=32,shuffle=True), DataLoader(testset)

def load_model():
    return Net().to(DEVICE)



if __name__ == "__main__":
    net = load_model()
    trainloader,testloader = load_data()
    train(net,trainloader,epochs=5,verbose=True)
    loss,accuracy = test(net,testloader=testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
