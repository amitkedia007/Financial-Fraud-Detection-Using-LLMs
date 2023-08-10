from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder


file_path = 'Final_Dataset.csv'
dataset = pd.read_csv(file_path)
text_data = dataset['Fillings'] 
labels = dataset['Fraud'] 
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
positive_label = label_encoder.transform(['yes'])[0]

X_temp, X_test, y_temp, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2 for validation

# Preprocess Text Data with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression Model:

logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Evaluate Model on Validation Set:

y_val_pred = logistic_model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, pos_label=positive_label)
val_recall = recall_score(y_val, y_val_pred, pos_label=positive_label)
val_f1 = f1_score(y_val, y_val_pred, pos_label=positive_label)

# Evaluate Model on Test Set:

y_test_pred = logistic_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=positive_label)
test_recall = recall_score(y_test, y_test_pred, pos_label=positive_label)
test_f1 = f1_score(y_test, y_test_pred, pos_label=positive_label)


print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1-score:", val_f1)

print("test Accuracy:", test_accuracy)
print("test Precision:", test_precision)
print("test Recall:", test_recall)
print("test F1-score:", test_f1)
