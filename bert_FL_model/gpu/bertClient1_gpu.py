import flwr as fl
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# ✅ Move computations to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# Define Model
class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(pooled_output), dim=1)  # ✅ Use softmax for multi-class classification

def load_custom_dataset(batch_size=16):
    train_data = torch.load("pth_data/dataset_train.pth")

    train_input_ids = train_data["input_ids"].to(device)
    train_attention_mask = train_data["attention_mask"].to(device)
    train_labels = train_data["labels"].to(device)  # Already one-hot encoded

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# Initialize Model
output_size = 16
model = DistilBERTClassifier(bert_model, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()  # ✅ Use CrossEntropyLoss

train_loader = load_custom_dataset()

# Define FL Client
class DistilBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, criterion, optimizer, epochs=1):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            torch.cuda.empty_cache()
            print("Starting training...")

            self.set_parameters(parameters)
            self.model.train()

            for epoch in range(self.epochs):
                for batch_idx, (input_ids, attention_mask, targets) in enumerate(self.train_loader):
                    print(f"Batch {batch_idx}: Input device {input_ids.device}, Model device {next(self.model.parameters()).device}")

                    # ✅ Ensure all tensors are on GPU
                    input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)

                    # ✅ Fix label encoding issue
                    loss = self.criterion(outputs, targets.argmax(dim=1))  # Convert one-hot targets to class indices

                    loss.backward()
                    self.optimizer.step()

                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            print("Training complete.")
            return self.get_parameters(config), len(self.train_loader.dataset), {}

        except Exception as e:
            print(f"🔥 Error during training: {e}")
            raise

    def evaluate(self, parameters, config):
        print("Starting evaluation...")
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, targets in self.train_loader:
                print(f"Evaluation batch: Input device {input_ids.device}, Model device {next(self.model.parameters()).device}")
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

                outputs = self.model(input_ids, attention_mask)
                loss += self.criterion(outputs, targets.argmax(dim=1)).item()
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                total += len(targets)

        accuracy = correct / total
        print(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(self.train_loader.dataset), {"accuracy": accuracy}

# Start Client
client = DistilBERTClient(model, train_loader, criterion, optimizer, epochs=1)
fl.client.start_client(server_address="localhost:9090", client=client.to_client())  # ✅ Use the updated client start function
