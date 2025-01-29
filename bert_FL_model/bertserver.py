import flwr as fl
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Define the model
class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return torch.sigmoid(self.classifier(pooled_output))  # Use sigmoid for multi-label tasks


# Load custom test data
def load_test_data(batch_size=16):
    test_data = torch.load("test_data.pth")
    test_input_ids = test_data["input_ids"]
    test_attention_mask = test_data["attention_mask"]
    test_labels = test_data["labels"]

    # Create test dataset and DataLoader
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# Evaluation function
def get_eval_fn(model):
    # Load the test dataset
    test_loader = load_test_data()

    def evaluate(server_round, parameters, config):
        print(f"Evaluating global model at round {server_round}")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        criterion = nn.BCELoss()
        loss = 0
        correct = 0
        total = 0

        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for input_ids, attention_mask, targets in test_loader:
                # Ensure targets have the correct shape [batch_size, num_classes]
                if targets.ndim > 2:
                    targets = targets.squeeze(-1)  # Remove extra dimensions if any

                # One-hot encode targets if necessary
                if targets.shape[1] != 16:  # If not already one-hot encoded
                    targets = F.one_hot(targets.to(torch.int64), num_classes=16).float()

                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss += criterion(outputs, targets).item()
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                total += len(targets)

                all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.argmax(dim=1).cpu().numpy())

        precision = precision_score(all_targets, all_outputs, average="weighted", zero_division=1)
        recall = recall_score(all_targets, all_outputs, average="weighted", zero_division=1)
        f1 = f1_score(all_targets, all_outputs, average="weighted", zero_division=1)
        accuracy = correct / total

        print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return loss / len(test_loader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    return evaluate


# Start the server
print("Loading tokenizer and initializing model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = DistilBERTClassifier(bert_model, output_size=16)

print("Setting up Flower server...")
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model)
)

print("Starting the Flower server...")
fl.server.start_server(server_address="localhost:9090", strategy=strategy, config=fl.server.ServerConfig(num_rounds=1))
