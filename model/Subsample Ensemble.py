import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split

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
X_train= np.concatenate((train_X_data1 ,train_X_data2,train_X_data3,train_X_data4), axis=1)
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
    SVC(kernel='linear', gamma=1.0, C=0.163, probability=True),
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191, max_depth=3, min_samples_leaf=3,
                               min_samples_split=20, random_state=0),
    RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0),
    DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
    ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0),
    lgb.LGBMClassifier(n_estimators=129, learning_rate=0.04392776161174538, max_depth=3, min_child_samples=20,
                       num_leaves=35, random_state=0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                       algorithm="SAMME", n_estimators=100, learning_rate=0.8),
    KNeighborsClassifier(n_neighbors=2),
    LogisticRegression(penalty='l2', solver='lbfgs', C=0.0012086328436524494, max_iter=100)
]

# Define subsample ensemble
subsample_predictions = []
# Create subsample ensemble
for clf in classifiers:
    # Create a random subsample of the training data
    X_train_subsample, _, y_train_subsample, _ = train_test_split(X_train_att, y_train, test_size=0.1, stratify=y_train)
    # Fit classifier on the subsample
    clf.fit(X_train_subsample, y_train_subsample)
    # Make predictions on the full test set
    subsample_predictions.append(clf.predict_proba(X_test_att)[:, 1])
# Calculate ensemble predictions using subsample ensemble
ensemble_prediction = np.mean(subsample_predictions, axis=0)
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
plt.savefig(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\AUC_compared.png')
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
