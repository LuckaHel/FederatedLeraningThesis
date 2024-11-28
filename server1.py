import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Define a perceptron model with two hidden layers
class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Create a test dataset (centralized, to be used on the server)
def create_test_dataloader(num_samples=100, input_size=30, batch_size=16):
    X = torch.rand(num_samples, input_size)
    y = (X.sum(axis=1) > 15).float().unsqueeze(1)  # Adjust threshold to match input size
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
        model = Perceptron(input_size=30, hidden_size1=32, hidden_size2=16, output_size=1)  # Match client model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Perform evaluation on the test dataset
        criterion = nn.BCELoss()
        loss = 0
        correct = 0
        total = 0

        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss += criterion(outputs, targets).item()  # Compute batch loss
                correct += (outputs.round() == targets).sum().item()  # Compute correct predictions
                total += len(targets)

                # Collect predictions and targets for calculating precision, recall, F1
                all_outputs.extend(outputs.round().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate precision, recall, and F1-score
        precision = precision_score(all_targets, all_outputs)
        recall = recall_score(all_targets, all_outputs)
        f1 = f1_score(all_targets, all_outputs)

        accuracy = correct / total
        avg_loss = loss / len(test_loader)

        # Return the average loss, accuracy, and additional metrics
        return avg_loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    return evaluate

# Define the federated learning strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients = 6,  # Minimum number of clients to participate in each round
    min_available_clients= 6,  # Minimum number of clients required to be available
    evaluate_fn=get_eval_fn()  # Attach the evaluation function
)

# Start the Flower server and run for a specified number of rounds
fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))

