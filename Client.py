# Download the model. 
# Local Training.
# Encrypting the updates with paillier scheme and use the public key of the server.
# Sending data updates to the proxy.
import sys
from collections import OrderedDict
from typing import List
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import Compose,ToTensor,Normalize
from torchvision import datasets

import flwr as fl
from flwr_datasets import FederatedDataset
from Model import test, train,load_model

def dataset_partitioner(dataset, batch_size, client_id, number_of_clients):
    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t CLIENT_ID
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler
    )
    return data_loader


def load_data(node_id):
    """Load partition MNIST data."""
    pytorch_transforms = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.EMNIST("./data",split="byclass",train=True,download=True,transform=pytorch_transforms)
    test_dataset = datasets.EMNIST("./data",split="byclass",train=False,download=True,transform=pytorch_transforms)
    # Divide data on each node: 80% train, 20% test
    

    
    trainloader = dataset_partitioner(dataset=train_dataset,batch_size=32,client_id=node_id,number_of_clients=200)
    testloader = dataset_partitioner(dataset=test_dataset,batch_size=32,client_id=node_id,number_of_clients=200)

    return trainloader, testloader

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)




    
node_id = 0
if len(sys.argv) > 1:
    node_id = int(sys.argv[1])

print("node : ", node_id)
net = load_model()
trainloader,testloader = load_data(node_id)
train_iter = iter(trainloader)
sample_images, sample_labels = next(train_iter)
print("Sample Batch Shape - Images:", sample_images.shape)
print("Sample Batch Shape - Labels:", sample_labels.shape)



fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(net=net,trainloader=trainloader,valloader=testloader),
)