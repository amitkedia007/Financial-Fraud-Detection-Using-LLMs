import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Assuming you've loaded the data into df
df = pd.read_csv('Final_Dataset.csv')

# Convert labels to binary
le = LabelEncoder()
df['Fraud'] = le.fit_transform(df['Fraud'])

# Split data into train, validation, and test sets
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['Fillings']).toarray()
X_val = vectorizer.transform(val['Fillings']).toarray()
X_test = vectorizer.transform(test['Fillings']).toarray()

y_train = train['Fraud'].values
y_val = val['Fraud'].values
y_test = test['Fraud'].values

# Define the ANN Model:
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hyperparameter Change 

# model = Sequential()

# # Input Layer
# model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.5))

# # Hidden Layer 1
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

# # Hidden Layer 2
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

# # Output Layer
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model = Sequential()

# Input Layer
model.add(Dense(256, activation='tanh', input_shape=(X_train.shape[1],)))

# Hidden Layer 1
model.add(Dense(128, activation='tanh'))

# Hidden Layer 2
model.add(Dense(64, activation='tanh'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



# Train the Model:

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop]
)

# Evaluate the Model:

# Validation performance
y_val_pred = (model.predict(X_val) > 0.5).astype("int32")
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Test performance
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-score: {val_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")

