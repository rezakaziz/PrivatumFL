# Receive encrypted updates
# Decrypt 
# Do the aggregation step.

import flwr as fl

strategy = fl.server.strategy.FedAvg(min_fit_clients=5,min_available_clients=5)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3),strategy=strategy)