import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset

# ✅ Load the training dataset
train_data = torch.load("train_data.pth")
train_input_ids = train_data["input_ids"]
train_attention_mask = train_data["attention_mask"]
train_labels = train_data["labels"]

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ✅ Load the testing dataset (for validation)
test_data = torch.load("test_data.pth")
test_input_ids = test_data["input_ids"]
test_attention_mask = test_data["attention_mask"]
test_labels = test_data["labels"]

test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("✅ Successfully loaded training and testing datasets!")

# ✅ Define the model
class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return torch.sigmoid(self.classifier(pooled_output))  # Sigmoid for multi-label classification


# ✅ Load BERT model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# ✅ Define model, optimizer, and loss function
output_size = train_labels.shape[1]  # Number of one-hot encoded classes
model = DistilBERTClassifier(bert_model, output_size)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label tasks

# ✅ Define Flower client with Early Stopping
class DistilBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, epochs=10, patience=3):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience  # Patience for early stopping
        self.best_loss = float("inf")  # Track best validation loss
        self.early_stop_counter = 0  # Track non-improving epochs

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Starting training...")
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.epochs):
            total_batches = len(self.train_loader)
            print(f"🔄 Epoch {epoch + 1}/{self.epochs} - Total Batches: {total_batches}")

            # Training loop
            train_loss = 0
            for batch_idx, (input_ids, attention_mask, targets) in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if batch_idx % 50 == 0:  # Print every 50 batches
                    print(f"📊 Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}")

            # Compute average training loss
            train_loss /= len(self.train_loader)

            # ✅ Validation (on test data)
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for input_ids, attention_mask, targets in self.test_loader:
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(self.test_loader)
            print(f"📉 Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # ✅ Early Stopping Logic
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stop_counter = 0  # Reset counter
                print("✅ New best model found! Saving...")
                torch.save(self.model.state_dict(), "best_model.pth")  # Save best model
            else:
                self.early_stop_counter += 1
                print(f"⚠️ No improvement for {self.early_stop_counter}/{self.patience} epochs.")

                if self.early_stop_counter >= self.patience:
                    print("⏹️ Early stopping triggered. Stopping training.")
                    break  # Stop training early

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("Starting evaluation...")
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            for input_ids, attention_mask, targets in self.test_loader:
                outputs = self.model(input_ids, attention_mask)
                loss += self.criterion(outputs, targets).item()
                correct += (outputs.round() == targets).sum().item()

        accuracy = correct / len(self.test_loader.dataset)
        print(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}

# ✅ Start the Flower client
client = DistilBERTClient(model, train_loader, test_loader, criterion, optimizer, epochs=1, patience=3)
fl.client.start_client(server_address="localhost:9090", client=client.to_client())
