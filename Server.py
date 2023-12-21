# Generate public and private keys
# Receive encrypted updates
# Decrypt 
# Do the aggregation step.

### Homomorphic encryption 
from phe import paillier
import json

from fedavg_he_dp import FedAvgModified

public_key, private_key = paillier.generate_paillier_keypair()

pub_json = json.dumps({'g': public_key.g, 'n':public_key.n})
with open("./Keys/public_key.json", "w") as outfile:
    outfile.write(pub_json)
######################################################

##### FL Server 
import flwr_modif as fl

strategy = FedAvgModified(min_fit_clients=2,min_available_clients=2,private_key=private_key)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3),strategy=strategy)