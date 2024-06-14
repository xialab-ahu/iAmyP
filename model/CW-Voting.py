import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Define Attention module
class Attention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(0, 1))
        attention_weights = torch.softmax(scores, dim=1)
        output = torch.matmul(attention_weights, V)
        return output

# Load datasets
train_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')

y_train = np.array(train_dataset['label'])
y_test = np.array(test_dataset['label'])

# Load features
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_morgan_40.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0,  delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2,header=0,  delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0, delimiter=',')
train_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_AAC.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0,  delimiter=',')
X_train = np.concatenate((train_X_data1 ,train_X_data2,train_X_data3,train_X_data4), axis=1)
test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_morgan_40.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0,  delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0,  delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0,delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_AAC.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0,delimiter=',')
X_test = np.concatenate((test_X_data1,test_X_data2,test_X_data3,test_X_data4), axis=1)

# Convert data to PyTorch tensors
train_Q = torch.tensor(X_train, dtype=torch.float32)
train_K = torch.tensor(X_train, dtype=torch.float32)
train_V = torch.tensor(X_train, dtype=torch.float32)
test_Q = torch.tensor(X_test, dtype=torch.float32)
test_K = torch.tensor(X_test, dtype=torch.float32)
test_V = torch.tensor(X_test, dtype=torch.float32)

# Create Attention model
attention_model = Attention()

# Calculate X_train and X_test
X_train_att = attention_model(train_Q, train_K, train_V).detach().numpy()
X_test_att = attention_model(test_Q, test_K, test_V).detach().numpy()

# Define classifiers
classifiers = [
    ('svc', SVC(kernel='linear', gamma=1.0, C=0.163, probability=True)),
    ('gbc', GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191, max_depth=3, min_samples_leaf=3, min_samples_split=20, random_state=0)),
    ('rf', RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0)),
    ('dt', DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5)),
    ('etc', ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=129, learning_rate=0.04392776161174538, max_depth=3, min_child_samples=20, num_leaves=35, random_state=0)),
    ('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=100, learning_rate=0.8)),
    ('knn', KNeighborsClassifier(n_neighbors=2)),
    ('lr', LogisticRegression(penalty='l2', solver='lbfgs', C=0.0012086328436524494, max_iter=100))
]

# Perform Stratified K-Fold Cross-Validation to calculate classifier weights
skf = StratifiedKFold(n_splits=5)
weights = []
for name, clf in classifiers:
    confidence_scores = []
    for train_index, val_index in skf.split(X_train_att, y_train):
        X_train_fold, X_val_fold = X_train_att[train_index], X_train_att[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        clf_clone = clone(clf)
        clf_clone.fit(X_train_fold, y_train_fold)
        pred_proba = clf_clone.predict_proba(X_val_fold)[:, 1]
        confidence_scores.extend(pred_proba.tolist())
    average_confidence = np.mean(confidence_scores) if confidence_scores else 0
    weights.append(average_confidence)

# Normalize weights
weights = np.array(weights)
weights /= np.sum(weights)

# Train classifiers and collect their weighted predictions
predictions = np.zeros((X_test_att.shape[0], len(classifiers)))
for idx, ((name, clf), weight) in enumerate(zip(classifiers, weights)):
    clf.fit(X_train_att, y_train)
    pred_proba = clf.predict_proba(X_test_att)[:, 1]
    predictions[:, idx] = pred_proba * weight

# Sum the weighted predictions and normalize
ensemble_prediction = np.sum(predictions, axis=1)
ensemble_prediction /= np.sum(weights)

# Calculate ensemble metrics
logloss = log_loss(y_test, ensemble_prediction)
accuracy = accuracy_score(y_test, ensemble_prediction > 0.5)
mcc = matthews_corrcoef(y_test, ensemble_prediction > 0.5)
precision = precision_score(y_test, ensemble_prediction > 0.5)
recall = recall_score(y_test, ensemble_prediction > 0.5)
f1 = f1_score(y_test, ensemble_prediction > 0.5)
confusion = confusion_matrix(y_test, ensemble_prediction > 0.5)

# Print metrics
print("Ensemble Log Loss:", logloss)
print("Ensemble Accuracy:", accuracy)
print("Ensemble MCC:", mcc)
print("Ensemble Precision:", precision)
print("Ensemble Recall:", recall)
print("Ensemble F1 Score:", f1)
# Print ensemble confusion matrix
print("Ensemble Confusion Matrix:\n", confusion)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, ensemble_prediction)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, ensemble_prediction)

# Calculate AUPR
aupr = auc(recall, precision)

# Print AUPR
print("Ensemble AUPR:", aupr)
