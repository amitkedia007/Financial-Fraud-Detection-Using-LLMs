import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Load the dataset
data = pd.read_csv('Final_Dataset.csv')

# Split data into train, validation, and test sets
train, temp = train_test_split(data, test_size=0.4, random_state=62)
val, test = train_test_split(temp, test_size=0.5, random_state=62)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train['Fillings'])
X_val = tfidf.transform(val['Fillings'])
X_test = tfidf.transform(test['Fillings'])

# Convert labels to numerical format
y_train = train['Fraud'].map({'no': 0, 'yes': 1}).values
y_val = val['Fraud'].map({'no': 0, 'yes': 1}).values
y_test = test['Fraud'].map({'no': 0, 'yes': 1}).values

# Initialize and train the XGBoost classifier
clf = xgb.XGBClassifier(
    reg_alpha=0.1,           # L1 regularization term on weight (increase for more regularization)
    reg_lambda=1,            # L2 regularization term on weight
    max_depth=3,             # Maximum depth of a tree (increase to make model more complex)
    min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
    learning_rate=0.3,       # Step size shrinkage used in update to prevents overfitting
)

eval_set = [(X_val, y_val)]
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

# Predictions
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

# Evaluation
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Precision:", precision_score(y_val, y_val_pred))
print("Validation Recall:", recall_score(y_val, y_val_pred))
print("Validation F1-score:", f1_score(y_val, y_val_pred))

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test F1-score:", f1_score(y_test, y_test_pred))
