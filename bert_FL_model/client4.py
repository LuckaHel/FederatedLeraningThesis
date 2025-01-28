from transformers import DistilBertTokenizer, DistilBertModel
import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset


# ✅ Load the dataset from processed_data_fixed.pth
data_path = "processed_data_fixed.pth"
data = torch.load(data_path) 
# ✅ Extract inputs and labels
input_ids = data["input_ids"]
attention_mask = data["attention_mask"]
labels = data["labels"]

# ✅ Ensure they are PyTorch tensors (convert if needed)
if not isinstance(input_ids, torch.Tensor):
    input_ids = torch.tensor(input_ids)
if not isinstance(attention_mask, torch.Tensor):
    attention_mask = torch.tensor(attention_mask)
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels)

# ✅ Create TensorDataset
dataset = TensorDataset(input_ids, attention_mask, labels)

# ✅ Define DataLoaders
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print("✅ Successfully loaded dataset from processed_data_fixed.pth!")


class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(DistilBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return torch.sigmoid(self.classifier(pooled_output))

# ✅ Load BERT model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# ✅ Define model, optimizer, and criterion
output_size = labels.shape[1]  # Number of classes in one-hot encoding
model = DistilBERTClassifier(bert_model, output_size)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()

# ✅ Define Flower client
class DistilBERTClient(fl.client.NumPyClient):
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
            self.model.train()

            for epoch in range(self.epochs):
                for batch_idx, (input_ids, attention_mask, targets) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

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
                outputs = self.model(input_ids, attention_mask)
                loss += self.criterion(outputs, targets).item()
                correct += (outputs.round() == targets).sum().item()

        accuracy = correct / len(self.test_loader.dataset)
        print(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}

# ✅ Start the Flower client
client = DistilBERTClient(model, train_loader, test_loader, criterion, optimizer, epochs=1)
fl.client.start_client(server_address="localhost:9090", client=client.to_client())

