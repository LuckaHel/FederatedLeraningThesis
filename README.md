# Federated Learning Thesis

This repository contains the implementation of a Federated Learning system using the Breast Cancer dataset. The system supports **3 clients and 1 server** by default, but the number of clients is configurable, ranging from a minimum of **2** to a maximum of **95**.

The model is based on a **2-layer perceptron**, and it utilizes **Federated Averaging (FedAvg)** for aggregating client models into a global model.

---

## 📋 Features
- **Dataset**: Breast Cancer dataset.
- **Clients**: Configurable (minimum 2, maximum 95).
- **Model Architecture**: 2-layer perceptron.
- **Aggregation Technique**: Federated Averaging (FedAvg).

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies
Ensure you have Python installed on your system. Then, install the required dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
## 🖥 How to Run the Code

### Running the Server
To start the server:
1. Open a terminal.
2. Navigate to the project directory.
3. Run the following command:
   ```bash
   python server1.py

### ⚙️ Configuration
Modifying Client Numbers

If you need to change the number of clients:

    Open the server1.py file in any text editor.
    Locate the section where the minimum and maximum number of clients are defined.
    Adjust these values to match your desired setup.

#### Notes for Running Clients

    The server must be running before starting any client scripts.
    Ensure the client IDs specified in runAllClients.py correspond to the actual client files (e.g., client1.py, client2.py).

### Running the Clients
**Option 1:** Run Individual Clients

To run a single client, use the command:
  ```bash
  python clientX.py


(Replace X with the client number, e.g., client1.py for Client 1).

**Option 2: **Run Multiple Clients

For convenience, use the provided runAllClients.py script to start multiple clients at once:

    Edit the runAllClients.py file to specify the clients you want to run.
    Execute the script:
  ```bash
  python runAllClients.py


This script will start the specified clients automatically.

Apologies for that! Here's the complete and fixed Markdown block for your README, ready for copy-pasting:

# Federated Learning Thesis

This repository contains the implementation of a Federated Learning system using the Breast Cancer dataset. The system supports **3 clients and 1 server** by default, but the number of clients is configurable, ranging from a minimum of **2** to a maximum of **95**.

The model is based on a **2-layer perceptron**, and it utilizes **Federated Averaging (FedAvg)** for aggregating client models into a global model.

---

## 📋 Features
- **Dataset**: Breast Cancer dataset.
- **Clients**: Configurable (minimum 2, maximum 95).
- **Model Architecture**: 2-layer perceptron.
- **Aggregation Technique**: Federated Averaging (FedAvg).

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies
Ensure you have Python installed on your system. Then, install the required dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt

### 2️⃣ Run the Server

Start the Federated Learning server by running:
```bash
python server1.py

### 3️⃣ Run the Clients

You can either:

    Run individual clients manually:
```bash
python clientX.py

(Replace X with the client number.)

Use the runAllClients.py script to start multiple clients:

    Open runAllClients.py and specify which clients to run.
    Execute the script:
```bash
        python runAllClients.py

## ⚙️ Configuration
Modifying Client Numbers

If you need to change the number of clients:

    Open the server1.py file in any text editor.
    Locate the section where the minimum and maximum number of clients are defined.
    Adjust these values to match your desired setup.

Notes for Running Clients

    The server must be running before starting any client scripts.
    Ensure the client IDs specified in runAllClients.py correspond to the actual client files (e.g., client1.py, client2.py).
the specified clients automatically.

## 🔍 Troubleshooting
**Common Issues**

    Server not detecting clients:
        Ensure the server is running before starting the clients.
        Verify that the client scripts are correctly named and correspond to the server's configuration.

    Dependency errors:
        Ensure all required packages are installed by running:
```bash
        pip install -r requirements.txt

    Client connections failing:
        Check your network configuration to ensure the server and clients can communicate.

