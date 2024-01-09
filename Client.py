"""
# Download the model. 
# Local Training.
# Encrypting the updates with paillier scheme and use the public key of the server.
# Sending data updates to the proxy.
"""



import json
import sys #for args handlings
### Imports for training the model with pytorch ###
from collections import OrderedDict
from typing import List
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler,TensorDataset,DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize
from torchvision import datasets
from torchsummary import summary
######################################################

### Imports for federated flower #####################
import flwr_modif as fl
from Model import test, train,load_model
######################################################

### Import Phe
from phe import paillier
import json 
with open('./Keys/public_key.json', 'r') as openfile:
 
    # Reading from json file
    pk = json.load(openfile)
 
public_key = paillier.PaillierPublicKey(n=int(pk['n']))
############################################
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

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

def load_femnist(writer_id):
    fileID = writer_id//100
    padding = writer_id%100


    # X Train and Y train
    fileName = './leaf/data/femnist/data/train/all_data_'+str(fileID)+'_niid_0_keep_0_train_9.json'
    file = open(fileName)
    data = json.load(file)
    num_samples = data['num_samples'][padding] 
    X_train = data['user_data'][data['users'][padding]]['x']
    X_train = np.array(X_train).reshape((num_samples,1,28,28)) 
    Y_train = data['user_data'][data['users'][padding]]['y']
    Y_train = np.array(Y_train)#.reshape((num_samples,1)) 
    file.close()
    Train_set = TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train))
    trainloader = DataLoader(Train_set,batch_size=32)
    # X test and Ytest 
    fileName = './leaf/data/femnist/data/test/all_data_'+str(fileID)+'_niid_0_keep_0_test_9.json'
    file = open(fileName)
    data = json.load(file)
    num_samples = data['num_samples'][padding] 
    X_test = data['user_data'][data['users'][padding]]['x']
    X_test = np.array(X_test).reshape((num_samples,1,28,28)) 
    Y_test = data['user_data'][data['users'][padding]]['y']
    Y_test = np.array(Y_test).reshape((num_samples)) 
    file.close()
    Test_set = TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test))
    valloader = DataLoader(Test_set,batch_size=32)
    print("UserID : ",data['users'][padding])
    print("Xtrain shape : ", X_train.shape,"Y_train shape : ",Y_train.shape)
    print("X_test shape : ", X_test.shape,"Y_test shape : ",Y_test.shape)

    return trainloader,valloader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        print(summary(self.net,(1,28,28),batch_size=32))
        self.trainloader = trainloader
        self.valloader = valloader
        

    def get_parameters(self, config):       
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        # multiply the parameters by len(self.trainloader)
        enc = get_encrypted_parameters(self.net)        
        return enc, len(self.trainloader),  {} # Need to modify this one 

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# add a function get_encrypted parameters
def get_encrypted_parameters(net):
    parameters = get_parameters(net)
    encrypted_parameters = []
    for parameter in parameters: 
            shape = parameter.shape
            encrypted_parameter = []
            
            for i in parameter.flatten():
                encrypted_parameter.append(public_key.encrypt(float(i)))
            encrypted_parameter = np.array(encrypted_parameter)
            encrypted_parameter = encrypted_parameter.reshape(shape)
            print("Encrypted Parameter shape ",encrypted_parameter.shape," Parameter shape",shape)
            encrypted_parameters.append(encrypted_parameter)
    return encrypted_parameters

def set_parameters(net, parameters: List[np.ndarray]):
    print(type(parameters))
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)




    
node_id = 1
if len(sys.argv) > 1:
    node_id = int(sys.argv[1])

print("node : ", node_id)
net = load_model()
trainloader,testloader = load_data(node_id)
#trainloader,testloader = load_femnist(node_id)



fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(net=net,trainloader=trainloader,valloader=testloader),
)
