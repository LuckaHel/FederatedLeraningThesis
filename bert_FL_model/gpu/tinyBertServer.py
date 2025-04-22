import flwr as fl
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
bert_model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)

class TinyBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(TinyBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(312, output_size)

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(pooled_output), dim=1)

def load_custom_dataset(batch_size=16):
    train_data = torch.load("pth_data/dataset_train.pth")
    dataset = TensorDataset(
        train_data["input_ids"].to(device),
        train_data["attention_mask"].to(device),
        train_data["labels"].to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TinyBERTClassifier(bert_model, output_size=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
train_loader = load_custom_dataset()

class TinyBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, criterion, optimizer, epochs=1):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v).to(device) for k, v in state_dict.items()}, strict=True)

    def fit(self, parameters, config):
        print("🧠 Training TinyBERT client...")
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.epochs):
            for batch_idx, (input_ids, attention_mask, targets) in enumerate(self.train_loader):
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets.argmax(dim=1))
                loss.backward()
                self.optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"[Epoch {epoch+1}] Batch {batch_idx+1} Loss: {loss.item():.4f}")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("📊 Evaluating TinyBERT client...")
        self.set_parameters(parameters)
        self.model.eval()

        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, targets in self.train_loader:
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                outputs = self.model(input_ids, attention_mask)
                loss += self.criterion(outputs, targets.argmax(dim=1)).item()
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                total += targets.size(0)

        accuracy = correct / total
        print(f"✅ Client Eval - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(self.train_loader.dataset), {"accuracy": accuracy}

client = TinyBERTClient(model, train_loader, criterion, optimizer)
fl.client.start_client(server_address="localhost:9090", client=client.to_client())

