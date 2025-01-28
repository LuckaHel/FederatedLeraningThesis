import flwr as fl
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from datasets import load_dataset

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


# Create a test dataset
def load_imdb_subset(tokenizer, batch_size=16, subset_size=100):
    dataset = load_dataset("imdb")
    train_texts = dataset["train"]["text"][:subset_size]
    train_labels = torch.tensor(dataset["train"]["label"][:subset_size], dtype=torch.float32).unsqueeze(1)
    test_texts = dataset["test"]["text"][:subset_size]
    test_labels = torch.tensor(dataset["test"]["label"][:subset_size], dtype=torch.float32).unsqueeze(1)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
    test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Evaluation function
def get_eval_fn(tokenizer, model):
    _, test_loader = load_imdb_subset(tokenizer)

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
                # Convert targets to one-hot encoded format
                # Ensure targets have the correct shape [batch_size, num_classes]
                targets = F.one_hot(targets.squeeze(-1).to(torch.int64), num_classes=16).float()


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
    min_fit_clients=2, min_available_clients=2, evaluate_fn=get_eval_fn(tokenizer, model)
)

print("Starting the Flower server...")
fl.server.start_server(server_address="localhost:9090", strategy=strategy, config=fl.server.ServerConfig(num_rounds=1))
