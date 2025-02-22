import flwr as fl
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ✅ Move computations to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        # ✅ Move input tensors to GPU
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(pooled_output), dim=1)  # ✅ Use softmax for multi-class classification

# Load Test Data
def load_test_data(batch_size=16):
    test_data = torch.load("test_data.pth")
    
    # ✅ Move test data to GPU
    test_input_ids = test_data["input_ids"].to(device)
    test_attention_mask = test_data["attention_mask"].to(device)
    test_labels = test_data["labels"].to(device)

    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Define Evaluation Function
def get_eval_fn(model):
    test_loader = load_test_data()

    def evaluate(server_round, parameters, config):
        print(f"Evaluating global model at round {server_round}")
        
        # ✅ Move model parameters to GPU before loading
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        loss, correct, total = 0, 0, 0
        all_targets, all_outputs = [], []

        with torch.no_grad():
            for input_ids, attention_mask, targets in test_loader:
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                outputs = model(input_ids, attention_mask)
                loss += criterion(outputs, targets.argmax(dim=1)).item()
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                total += len(targets)

                all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.argmax(dim=1).cpu().numpy())

        precision = precision_score(all_targets, all_outputs, average="weighted", zero_division=1)
        recall = recall_score(all_targets, all_outputs, average="weighted", zero_division=1)
        f1 = f1_score(all_targets, all_outputs, average="weighted", zero_division=1)
        accuracy = correct / total

        print(f"Eval - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return loss / len(test_loader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    return evaluate

# Initialize and Start Server
print("Loading tokenizer and initializing model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)  # ✅ Move model to GPU
model = DistilBERTClassifier(bert_model, output_size=16).to(device)

print("Setting up Flower server...")
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model)
)

print("Starting the Flower server...")
fl.server.start_server(server_address="localhost:9090", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))
