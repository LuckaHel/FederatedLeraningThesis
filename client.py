import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer (hidden layer to output)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # Pass through the second layer with sigmoid activation (since it's binary classification)
        x = torch.sigmoid(self.fc2(x))
        return x
# Define a Flower client
class PerceptronClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def get_parameters(self, config):
        # Detach the tensor before calling .numpy() to prevent the grad error
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Debugging: Check training dataset size
        print(f"Training dataset size: {len(self.train_loader.dataset)}")
        
        for epoch in range(1):  # Training for 1 epoch (adjust if needed)
            for inputs, targets in self.train_loader:
                # Debugging: Check input and target shapes
                print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
        # Debugging: Check parameters after training
        print(f"Updated model parameters: {self.get_parameters(config)}")
        
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("Evaluate function is called")
        self.set_parameters(parameters)
        self.model.eval()

        # Debugging: Check if evaluation is starting
        print(f"Starting evaluation...")

        loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Debugging: Check if we are getting inputs and targets
                print(f"Evaluating batch with inputs shape: {inputs.shape} and targets shape: {targets.shape}")
                
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets).item()
                loss += batch_loss
                
                # Debugging: Check model outputs and loss
                print(f"Model outputs: {outputs}, Batch loss: {batch_loss}")
                
                correct += (outputs.round() == targets).sum().item()
    
        accuracy = correct / len(self.test_loader.dataset)
    
        # Debugging: Check final evaluation results
        print(f"Final loss: {loss}, Final accuracy: {accuracy}")
    
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}


# Create random data for two clients
def create_dataloader(num_samples=100, input_size=2, batch_size=16):
    X = torch.rand(num_samples, input_size)
    y = (X.sum(axis=1) > 1).float().unsqueeze(1)  # Binary classification
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup data loaders, model, and start Flower client
input_size = 2
hidden_size = 4  # You can adjust this to experiment with different hidden layer sizes
output_size = 1
model = Perceptron(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

train_loader = create_dataloader()
test_loader = create_dataloader()

# Debugging: Check the train and test dataset sizes
print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

client = PerceptronClient(model, train_loader, test_loader, criterion, optimizer)
# Test evaluate() manually before starting the Flower client
loss, dataset_size, metrics = client.evaluate(client.get_parameters(config=None), config=None)
print(f"Manual test - Loss: {loss}, Dataset Size: {dataset_size}, Metrics: {metrics}")

# Convert the NumPyClient to a Client and start it
fl.client.start_client(server_address="localhost:8080", client=client.to_client())

