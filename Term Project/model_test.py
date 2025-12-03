# Importing the model
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
import os
import pandas as pd

base_dir = os.path.dirname(__file__)  # directory of the script
file_path = os.path.join(base_dir, 'data', 'dataset2', 'X_test_processed.csv')
file_path_y = os.path.join(base_dir, 'data', 'dataset2', 'y_test.csv')
model_path = os.path.join(base_dir, 'lr.pkl')

X_test_1 = pd.read_csv(file_path)
y_test_1 = pd.read_csv(file_path_y)
# Load model from pkl file
with open(model_path, "rb") as f:
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
print("Entire Metrics:")
print(classification_report(y_test_1, y_pred_1))