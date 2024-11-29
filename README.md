# FederatedLeraningThesis
This repository contains the code for a Federated Learning system that operates with the Breast Cancer dataset. The current implementation includes 3 clients and 1 server, with configurable client numbers ranging from a minimum of 2 to a maximum of 95.

The system utilizes a 2-layer perceptron for model training and Federated Averaging (FedAvg) to aggregate the client models.
Features

    Dataset: Breast Cancer dataset.
    Clients: Configurable number of clients (minimum 2, maximum 95).
    Model: 2-layer perceptron.
    Aggregation: Federated Averaging.

Requirements

To run this project, ensure all dependencies listed in requirements.txt are installed. This project is compatible with Linux-based systems.
Usage Instructions
Step 1: Install Dependencies

pip install -r requirements.txt

Step 2: Start the Server

Run the server script:

python server1.py

Step 3: Start the Clients

You can either run clients individually:

python clientX.py

(Replace X with the client number.)

Or, use the provided runAllClients.py script to start multiple clients:

    Edit runAllClients.py to specify which clients to run.
    Execute the script:

    python runAllClients.py

Notes

    Configuring Client Numbers:
        Update server1.py to adjust the minimum and maximum number of clients in the designated lines.

    Running the Clients:
        The server must be running before starting any clients.

