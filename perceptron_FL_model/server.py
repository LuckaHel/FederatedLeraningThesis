import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define a perceptron model with an additional hidden layer
class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer (input to hidden)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second layer (hidden to output)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU after first layer
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid after second layer for binary classification
        return x

# Create a test dataset (centralized, to be used on the server)
def create_test_dataloader(num_samples=100, input_size=2, batch_size=16):
    X = torch.rand(num_samples, input_size)
    y = (X.sum(axis=1) > 1).float().unsqueeze(1)  # Binary classification
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define an evaluation function for centralized evaluation
def get_eval_fn():
    """Return an evaluation function for centralized evaluation."""

    # Initialize the test dataset on the server
    test_loader = create_test_dataloader()

    # Define the evaluation logic
    def evaluate(server_round, parameters, config):
        print(f"Evaluating global model at round {server_round}")

        # Initialize the model and set its parameters from the server
        model = Perceptron(input_size=30, hidden_size=4, output_size=1)  # Match client model architecture
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Perform evaluation on the test dataset
        criterion = nn.BCELoss()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss += criterion(outputs, targets).item()  # Compute batch loss
                correct += (outputs.round() == targets).sum().item()  # Compute correct predictions
                total += len(targets)

        accuracy = correct / total
        avg_loss = loss / len(test_loader)

        # Return the average loss and accuracy
        return avg_loss, {"accuracy": accuracy}

    return evaluate

# Define the federated learning strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,  # Minimum number of clients to participate in each round
    min_available_clients=2,  # Minimum number of clients required to be available
    evaluate_fn=get_eval_fn()  # Attach the evaluation function
)

# Start the Flower server and run for a specified number of rounds
fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))

