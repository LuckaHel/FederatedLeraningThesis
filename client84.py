from sklearn.datasets import load_breast_cancer
from torch.utils.data import DataLoader, TensorDataset
import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # Corrected to hidden_size1
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class PerceptronClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.epochs):
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss += self.criterion(outputs, targets).item()
                correct += (outputs.round() == targets).sum().item()

        accuracy = correct / len(self.test_loader.dataset)
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}

def load_breast_cancer_data(batch_size=16):
    data = load_breast_cancer()
    X = torch.tensor(data['data'], dtype=torch.float32)
    y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)  # Binary targets
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test split
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Model setup
input_size = 30  # Features in the breast cancer dataset
hidden_size1 = 32  # Can be adjusted
hidden_size2 =16
output_size = 1  
# Binary classification output
model = Perceptron(input_size, hidden_size1, hidden_size2, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Load the breast cancer dataset
train_loader, test_loader = load_breast_cancer_data()

# Instantiate the client and start federated learning
client = PerceptronClient(model, train_loader, test_loader, criterion, optimizer, epochs=3)
fl.client.start_client(server_address="localhost:8080", client=client)

