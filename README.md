# PrivatumFL

**PrivatumFL** is an experimental implementation of **Federated Learning (FL)** in Python, extended with **privacy and security mechanisms** such as **Differential Privacy (DP)**, **encryption**, and **secure aggregation**.

---

## Table of Contents

- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Run the Server](#run-the-server)  
  - [Run a Client](#run-a-client)  

---

## Overview

**PrivatumFL** provides a server–client architecture for training machine learning models across multiple devices while ensuring **data privacy**.  
It includes custom modifications of the **Flower (flwr)** framework to support:

- **Differential Privacy** integration (`FedAvgModified`)  
- **Homomorphic Encryption** to protect Gradients from the server  

---


## Project Structure
```text
PrivatumFL/
├── Client.py          # Federated learning client code
├── Server.py          # Central server code
├── Model.py           # PyTorch model definitions
├── fedavg_he_dp.py    # FedAvg with DP and encryption
├── flwr_modif/        # Modified Flower framework
│   └── ...            # Submodules (client, server, common, proto, etc.)
├── Keys/              # Keys/secrets storage for HE
└── .gitignore
```

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rezakaziz/PrivatumFL.git
   cd PrivatumFL
   ```
2. Create and activate a virtual environment:
  ```
   python3 -m venv venv
  source venv/bin/activate      
  ```
3. Install dependencies:
   ```
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
    ```
## Usage

### Run the Server
Run the central server that coordinates federated training:
  ```
  python Server.py 
  ```
**Remark:** To configure the number of clients expected to connect, you need to edit the file Server.py directly.

### Run a Client
Run one or more clients that participate in training:
  ```
  python Client.py <node_id>
  ```
