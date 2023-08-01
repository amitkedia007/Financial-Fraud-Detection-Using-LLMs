from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label)}

    def __len__(self):
        return len(self.texts)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load the data
df = pd.read_csv('Final_Dataset.csv')

# Concatenate the section texts to form the input data
input_data = df[['section_1', 'section_1A', 'section_1B', 'section_2', 'section_3', 'section_4', 'section_5', 'section_9B', 'section_10', 'section_11', 'section_12', 'section_13', 'section_14', 'section_15']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Use the "Fraud" column as the target data
# Convert 'Yes'/'No' to 1/0
target_data = df['Fraud'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_data, target_data, test_size=0.2)



# Load pretrained model and tokenizer

# Load pretrained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Check if padding token is set, if not, set it
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer) + 1)
model.config.pad_token_id = tokenizer.pad_token_id


# Create the datasets
train_dataset = FraudDataset(input_train.tolist(), target_train.tolist(), tokenizer)
val_dataset = FraudDataset(input_val.tolist(), target_val.tolist(), tokenizer)

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

# Create the Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

metrics = trainer.evaluate()
# Save the model
model.save_pretrained('./saved_model')

print(metrics)