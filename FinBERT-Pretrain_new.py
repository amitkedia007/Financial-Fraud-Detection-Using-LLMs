# Import Libraries
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Load Dataset
file_path = 'Final_Dataset.csv'
dataset = pd.read_csv(file_path)

# Encode Labels
label_encoder = LabelEncoder()
dataset['Fraud'] = label_encoder.fit_transform(dataset['Fraud'])

# Tokenize the Text Data
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
text_data = dataset['Fillings'].tolist()
max_length = 512
encoded_inputs = tokenizer(text_data, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
attention_masks = encoded_inputs['attention_mask']

# Split the Dataset into Training, Validation, and Test Sets
train_inputs, temp_inputs, train_labels, temp_labels, train_masks, temp_masks = train_test_split(
    encoded_inputs['input_ids'], dataset['Fraud'], attention_masks, test_size=0.4, random_state=42
)

val_inputs, test_inputs, val_labels, test_labels, val_masks, test_masks = train_test_split(
    temp_inputs, temp_labels, temp_masks, test_size=0.5, random_state=42
)

# Create DataLoader with attention masks
train_dataset = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels.values).long())
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_masks, torch.tensor(val_labels.values).long())
val_dataloader = DataLoader(val_dataset, batch_size=32)

test_dataset = TensorDataset(test_inputs, test_masks, torch.tensor(test_labels.values).long())
test_dataloader = DataLoader(test_dataset, batch_size=32)


# Load Pre-trained BERT Model for Sequence Classification
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain', num_labels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training Loop
model.train()
for epoch in range(3):
    for batch in train_dataloader:
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# Evaluation
model.eval()
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        test_predictions.extend(predictions.tolist())
        test_true_labels.extend(labels.tolist())

# Metrics for Test Set
accuracy = accuracy_score(test_true_labels, test_predictions)
precision = precision_score(test_true_labels, test_predictions)
recall = recall_score(test_true_labels, test_predictions)
f1 = f1_score(test_true_labels, test_predictions)
roc_auc = roc_auc_score(test_true_labels, test_predictions)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1)
print("Test ROC-AUC:", roc_auc)
