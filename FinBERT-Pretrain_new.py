# Import Libraries
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


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

print(tokenizer)

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
    start_time = time.time()   # track start time
    for batch in train_dataloader:
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        end_time = time.time()   #track end time
        epoch_time = end_time - start_time   # Time taken to train the model
    
        print(f"Time taken for epoch {epoch+1}: {epoch_time:.2f} seconds")

total_end_time = time.time()
total_time = total_end_time - start_time

print(f"Total training time: {total_time:.2f} seconds")
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

# Training Loop with attention masks
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_predictions = []
    train_true_labels = []
    for batch in train_dataloader:
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        train_predictions.extend(predictions.tolist())
        train_true_labels.extend(labels.tolist())
    
    # Training Metrics
    train_accuracy = accuracy_score(train_true_labels, train_predictions)
    train_precision = precision_score(train_true_labels, train_predictions)
    train_recall = recall_score(train_true_labels, train_predictions)
    train_f1 = f1_score(train_true_labels, train_predictions)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {train_loss/len(train_dataloader)}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Training Precision: {train_precision}")
    print(f"Training Recall: {train_recall}")
    print(f"Training F1-score: {train_f1}")
    
    # Validation Loop
    model.eval()
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, masks, labels = batch
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
            val_predictions.extend(predictions.tolist())
            val_true_labels.extend(labels.tolist())
    
    # Validation Metrics
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_precision = precision_score(val_true_labels, val_predictions)
    val_recall = recall_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions)

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1-score: {val_f1}\n")

#1. Plot the Confusion Matrix
cm = confusion_matrix(test_true_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraudulent", "Fraudulent"])
disp.plot()
plt.title('Confusion Matrix for Test Data')
plt.show()

# 2. Plot the ROC Curve
fpr, tpr, _ = roc_curve(test_true_labels, test_predictions)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='BERT Model')
roc_display.plot()
plt.title('ROC Curve for Test Data')
plt.show()

# 3. Plot the Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(test_true_labels, test_predictions)
pr_display = PrecisionRecallDisplay(precision=precision_curve, recall=recall_curve)
pr_display.plot()
plt.title('Precision-Recall Curve for Test Data')
plt.show()