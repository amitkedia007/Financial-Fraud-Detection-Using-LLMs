import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load the data
df = pd.read_csv('Final_Dataset.csv')



# Combine all the section columns into a single text column
section_columns = [col for col in df.columns if 'section' in col]
df['text'] = df[section_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Convert the labels to 0 and 1
df['label'] = df['Fraud'].map({'No': 0, 'Yes': 1})

df = df[['text', 'label']]

# Split the data into train and test
train_df, test_df = train_test_split(df, test_size=0.2)

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-pretrain", num_labels=2)

# Tokenize the text data for training and testing
train_encodings = tokenizer.batch_encode_plus(train_df['text'].tolist(), padding='max_length', truncation=True, max_length=512)
test_encodings = tokenizer.batch_encode_plus(test_df['text'].tolist(), padding='max_length', truncation=True, max_length=512)

# Create a PyTorch Dataset for training and testing
class FraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FraudDataset(train_encodings, train_df['label'].tolist())
test_dataset = FraudDataset(test_encodings, test_df['label'].tolist())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()

# Print the evaluation results
print(eval_results)


print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

print(f"Training set label distribution:\n{train_df['label'].value_counts(normalize=True)}")
print(f"Test set label distribution:\n{test_df['label'].value_counts(normalize=True)}")

common_indices = train_df.index.intersection(test_df.index)
print(f"Number of common indices: {len(common_indices)}")
