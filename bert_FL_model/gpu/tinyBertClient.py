import flwr as fl
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TinyBERT Classifier
class TinyBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(TinyBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(312, output_size)  # TinyBERT hidden size is 312

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(pooled_output), dim=1)

val_loss_per_round = {}
val_acc_per_round = {}

def get_eval_fn(model):
    def evaluate(server_round, parameters, config):
        print(f"\n🌐 Evaluating TinyBERT model at round {server_round}")
        state_dict = {k: torch.tensor(v).to(device) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict, strict=True)

        val_data = torch.load("pth_data/dataset_validation.pth")
        val_loader = DataLoader(
            TensorDataset(val_data["input_ids"].to(device),
                          val_data["attention_mask"].to(device),
                          val_data["labels"].to(device)),
            batch_size=16, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        model.eval()

        batch_losses, batch_accuracies = [], []
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, targets) in enumerate(val_loader):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets.argmax(dim=1)).item()
                acc = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item() / len(targets)

                batch_losses.append(loss)
                batch_accuracies.append(acc)
                all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.argmax(dim=1).cpu().numpy())

        val_loss_per_round[server_round] = batch_losses
        val_acc_per_round[server_round] = batch_accuracies

        mean_loss = sum(batch_losses) / len(batch_losses)
        mean_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        precision = precision_score(all_targets, all_outputs, average="weighted", zero_division=1)
        recall = recall_score(all_targets, all_outputs, average="weighted", zero_division=1)
        f1 = f1_score(all_targets, all_outputs, average="weighted", zero_division=1)

        print(f"✅ Round {server_round} Summary | Loss: {mean_loss:.4f}, Acc: {mean_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return mean_loss, {
            "accuracy": mean_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "val_loss_batches": batch_losses,
            "val_acc_batches": batch_accuracies,
        }

    return evaluate

print("🚀 Loading TinyBERT...")
tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
bert_model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)
model = TinyBERTClassifier(bert_model, output_size=16).to(device)

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model)
)

print("🌸 Starting Flower server...")
fl.server.start_server(
    server_address="localhost:9090",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=5)
)

