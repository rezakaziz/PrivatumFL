# Receive encrypted updates
# Decrypt 
# Do the aggregation step.

import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))