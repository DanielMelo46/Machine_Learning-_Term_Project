# Importing the model
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score

X_test_1 = pd.read_csv('data/dataset2/X_test_processed.csv')
y_test_1 = pd.read_csv('data/dataset2/y_test.csv')
# Load model from pkl file
with open("lr.pkl", "rb") as f:
    model = pickle.load(f)

# Use the model
test_results_1 = []
y_pred_1 = model.predict(X_test_1)
y_prob_test_1 = model.predict_proba(X_test_1)[:, 1]

test_results_1.append({
        'Accuracy': accuracy_score(y_test_1, y_pred_1),
        'Recall': recall_score(y_test_1, y_pred_1),
        'F1 Score': f1_score(y_test_1, y_pred_1),
        'ROC-AUC': roc_auc_score(y_test_1, y_prob_test_1)
})

print(pd.DataFrame(test_results_1))