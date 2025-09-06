import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_absolute_percentage_error
)

# ======================
# Example: Model Testing
# ======================

# Suppose you have test data (X_test, y_test) and a trained PyTorch model
# Replace this with your actual test dataset
X_test = torch.randn(100, 10)   # dummy test data (100 samples, 10 features)
y_test = np.random.randint(0, 2, size=100)  # dummy true labels (binary)

# Forward pass through your trained model
# Example: model is a torch.nn.Module (replace with yours)
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = DummyModel()
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred = torch.argmax(y_pred_probs, dim=1).numpy()

# ======================
# Classification Metrics
# ======================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ======================
# Confusion Matrix Plot
# ======================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ======================
# ROC Curve & AUC
# ======================
y_pred_proba = torch.softmax(y_pred_probs, dim=1)[:, 1].numpy()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ======================
# MAPE (if regression-like metric is required)
# ======================
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)
