from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Load and preprocess a subset of the dataset
def load_dataset_subset(tokenizer, batch_size=16, subset_size=100):  
    dataset = load_dataset("imdb")  # Change if using a different dataset
    train_texts = dataset["train"]["text"][:subset_size]
    train_labels = torch.tensor(dataset["train"]["label"][:subset_size], dtype=torch.int64)
    test_texts = dataset["test"]["text"][:subset_size]
    test_labels = torch.tensor(dataset["test"]["label"][:subset_size], dtype=torch.int64)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Convert labels to one-hot encoding for 16-class classification
    train_labels = F.one_hot(train_labels, num_classes=16).float()
    test_labels = F.one_hot(test_labels, num_classes=16).float()

    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
    test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Flower client
class TinyBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, epochs=1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            print("Starting training...")
            self.set_parameters(parameters)
            self.model.train()  # Ensure model is in training mode

            for epoch in range(self.epochs):
                for batch_idx, (input_ids, attention_mask, targets) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask).logits  # Use .logits for classification

                    # Apply sigmoid to outputs before BCELoss
                    loss = self.criterion(torch.sigmoid(outputs), targets)
                    loss.backward()
                    self.optimizer.step()

                    # Log every 10 batches
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            print("Training complete.")
            return self.get_parameters(config), len(self.train_loader.dataset), {}

        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def evaluate(self, parameters, config):
        print("Starting evaluation...")
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            for input_ids, attention_mask, targets in self.test_loader:
                outputs = self.model(input_ids, attention_mask).logits  # Use .logits for classification
                loss += self.criterion(torch.sigmoid(outputs), targets).item()
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        accuracy = correct / len(self.test_loader.dataset)
        print(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D", num_labels=16
)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()

# Load dataset
train_loader, test_loader = load_dataset_subset(tokenizer)

# Initialize and start the client
client = TinyBERTClient(model, train_loader, test_loader, criterion, optimizer, epochs=1)
fl.client.start_client(server_address="localhost:9090", client=client.to_client())
