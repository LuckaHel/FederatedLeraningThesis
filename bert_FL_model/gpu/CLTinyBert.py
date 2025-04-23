import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load datasets
train_data = torch.load("pth_data/dataset_train.pth")
val_data = torch.load("pth_data/dataset_validation.pth")
test_data = torch.load("pth_data/dataset_test.pth")

# Create DataLoaders
train_dataset = TensorDataset(train_data["input_ids"], train_data["attention_mask"], train_data["labels"])
val_dataset = TensorDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])
test_dataset = TensorDataset(test_data["input_ids"], test_data["attention_mask"], test_data["labels"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load TinyBERT model
model_name = "huawei-noah/TinyBERT_General_4L_312D"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    print(f"\n🚀 Epoch {epoch+1}/5")
    total_loss = 0.0
    model.train()
    loop = tqdm(train_loader, desc="Training", leave=False)

    for step, batch in enumerate(loop):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} — Avg Loss: {total_loss/len(train_loader):.4f}")

    # Validation metrics
    print("📊 Validating...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    print(f"✅ Validation — Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# Final test evaluation
print("\n🏁 Final Test Set Evaluation:")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.argmax(dim=1).cpu().numpy())

print(classification_report(all_labels, all_preds))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Optional: MBTI class labels
labels = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
          'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save the fine-tuned model and tokenizer
model.save_pretrained("finetuned_tinybert")
tokenizer.save_pretrained("finetuned_tinybert")

