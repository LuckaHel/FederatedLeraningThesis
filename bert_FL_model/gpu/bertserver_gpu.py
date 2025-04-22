import flwr as fl
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(pooled_output), dim=1)

# Load Test Data (optional, still used for final evaluation)
def load_test_data(batch_size=16):
    test_data = torch.load("pth_data/dataset_test.pth")
    dataset = TensorDataset(
        test_data["input_ids"].to(device),
        test_data["attention_mask"].to(device),
        test_data["labels"].to(device),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# GLOBAL STORAGE for validation tracking
val_loss_per_round = {}
val_acc_per_round = {}

# Evaluation function factory
def get_eval_fn(model):
    def evaluate(server_round, parameters, config):
        print(f"\n🌐 Evaluating global model at round {server_round}")
        state_dict = {k: torch.tensor(v).to(device) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict, strict=True)

        # Load validation set
        val_data = torch.load("pth_data/dataset_validation.pth")
        input_ids = val_data["input_ids"].to(device)
        attention_mask = val_data["attention_mask"].to(device)
        labels = val_data["labels"].to(device)

        val_dataset = TensorDataset(input_ids, attention_mask, labels)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        model.eval()

        batch_losses = []
        batch_accuracies = []
        all_targets, all_outputs = [], []

        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, targets) in enumerate(val_loader):
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                outputs = model(input_ids, attention_mask)

                loss = criterion(outputs, targets.argmax(dim=1)).item()
                acc = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item() / len(targets)

                batch_losses.append(loss)
                batch_accuracies.append(acc)

                all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.argmax(dim=1).cpu().numpy())

        # Store batch-level metrics
        val_loss_per_round[server_round] = batch_losses
        val_acc_per_round[server_round] = batch_accuracies

        # Compute aggregate metrics
        mean_loss = sum(batch_losses) / len(batch_losses)
        mean_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        precision = precision_score(all_targets, all_outputs, average="weighted", zero_division=1)
        recall = recall_score(all_targets, all_outputs, average="weighted", zero_division=1)
        f1 = f1_score(all_targets, all_outputs, average="weighted", zero_division=1)

        print(f"✅ Validation Round {server_round} Summary")
        print(f"Avg Loss: {mean_loss:.4f}, Avg Acc: {mean_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # You could return val_loss_per_round and val_acc_per_round separately if needed
        return mean_loss, {
            "accuracy": mean_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "val_loss_batches": batch_losses,
            "val_acc_batches": batch_accuracies,
        }

    return evaluate

# 🌱 Initialize & Start Flower Server
print("🚀 Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model = DistilBERTClassifier(bert_model, output_size=16).to(device)

print("🌸 Starting Flower server...")
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model),
)

fl.server.start_server(
    server_address="localhost:9090",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=5)
)
