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
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix


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

# Define and Train the Model (ReLU Activation)
model_relu = Sequential()
model_relu.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model_relu.add(Dense(128, activation='relu'))
model_relu.add(Dense(64, activation='relu'))
model_relu.add(Dense(1, activation='sigmoid'))
model_relu.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history_relu = model_relu.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop]
)

# Define and Train the Model (Tanh Activation)
model_tanh = Sequential()
model_tanh.add(Dense(256, activation='tanh', input_shape=(X_train.shape[1],)))
model_tanh.add(Dense(128, activation='tanh'))
model_tanh.add(Dense(64, activation='tanh'))
model_tanh.add(Dense(1, activation='sigmoid'))
model_tanh.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history_tanh = model_tanh.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop]
)

# Evaluate and Plot Graphs
def evaluate_and_plot(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraudulent", "Fraudulent"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Plot ROC Curve
    y_probas = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probas)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Model')
    roc_display.plot()
    plt.title('ROC Curve')
    plt.show()

    # Plot Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probas)
    pr_display = PrecisionRecallDisplay(precision=precision_curve, recall=recall_curve)
    pr_display.plot()
    plt.title('Precision-Recall Curve')
    plt.show()

print("Results for ReLU Activation:")
evaluate_and_plot(model_relu, X_test, y_test)

print("Results for Tanh Activation:")
evaluate_and_plot(model_tanh, X_test, y_test)
