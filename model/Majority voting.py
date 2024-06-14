import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn import metrics

# Load your existing features and labels
train_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')

y_train = train_dataset['label'].values
y_test = test_dataset['label'].values

# Load your existing features
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_morgan_40.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0, delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2, header=0, delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0, delimiter=',')
train_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_AAC.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0, delimiter=',')
X_train_data = np.concatenate((train_X_data1, train_X_data2, train_X_data3, train_X_data4), axis=1)

test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_morgan_40.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0, delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0, delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0, delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_AAC.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0, delimiter=',')
X_test_data = np.concatenate((test_X_data1, test_X_data2, test_X_data3, test_X_data4), axis=1)

# Convert data to PyTorch tensors
train_Q = torch.tensor(X_train_data, dtype=torch.float32)
train_K = torch.tensor(X_train_data, dtype=torch.float32)
train_V = torch.tensor(X_train_data, dtype=torch.float32)
test_Q = torch.tensor(X_test_data, dtype=torch.float32)
test_K = torch.tensor(X_test_data, dtype=torch.float32)
test_V = torch.tensor(X_test_data, dtype=torch.float32)

# Define Attention module
class Attention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(0, 1))
        attention_weights = torch.softmax(scores, dim=1)
        output = torch.matmul(attention_weights, V)
        return output

# Create Attention model
attention_model = Attention()

# Calculate X_train and X_test
X_train= attention_model(train_Q, train_K, train_V)
X_test = attention_model(test_Q, test_K, test_V)

# Define classifiers
clf_svm = SVC(kernel='linear', gamma=1.0, C=0.163, probability=True)
clf_gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191,
                                      max_depth=3, min_samples_leaf=3, min_samples_split=20, random_state=0)
clf_rf = RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2,
                                min_samples_leaf=1, random_state=0)
clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5)
clf_ert = ExtraTreesClassifier(n_estimators=391, max_depth=10, min_samples_split=10, min_samples_leaf=1, random_state=0)
clf_lightgbm = lgb.LGBMClassifier(n_estimators=129, learning_rate=0.04392776161174538, max_depth=3,
                                 min_child_samples=20, num_leaves=35, random_state=0)
clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME", n_estimators=1000, learning_rate=0.8)
clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.0012086328436524494, max_iter=1000)

# Train classifiers
clf_svm.fit(X_train, y_train)
clf_gbdt.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)
clf_dt.fit(X_train, y_train)
clf_ert.fit(X_train, y_train)
clf_lightgbm.fit(X_train, y_train)
clf_ada.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)
clf_lr.fit(X_train, y_train)

# Predictions
y_prob_svm = clf_svm.predict_proba(X_test)[:, 1]
y_prob_gbdt = clf_gbdt.predict_proba(X_test)[:, 1]
y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
y_prob_dt = clf_dt.predict_proba(X_test)[:, 1]
y_prob_ert = clf_ert.predict_proba(X_test)[:, 1]
y_prob_lightgbm = clf_lightgbm.predict_proba(X_test)[:, 1]
y_prob_ada = clf_ada.predict_proba(X_test)[:, 1]
y_prob_knn = clf_knn.predict_proba(X_test)[:, 1]
y_prob_lr = clf_lr.predict_proba(X_test)[:, 1]

classifiers = [clf_svm, clf_gbdt, clf_rf, clf_dt, clf_ert, clf_lightgbm, clf_ada, clf_knn, clf_lr]

# Fit classifiers and get predicted probabilities
predictions = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    predictions.append(y_pred_prob)

# Majority Voting
predictions = np.array(predictions)
print(predictions)
y_pred_majority = np.round(np.mean(predictions, axis=0))  # Voting based on the mean probability
print(y_pred_majority)
# Evaluate performance
accuracy_majority = accuracy_score(y_test, y_pred_majority)
aupr_majority = average_precision_score(y_test, np.mean(predictions, axis=0))
roc_auc_majority = metrics.roc_auc_score(y_test,  np.mean(predictions, axis=0))
mcc_voting = matthews_corrcoef(y_test, y_pred_majority)
print(f"Majority Voting MCC: {mcc_voting}")


# Other performance metrics
precision_majority = precision_score(y_test, y_pred_majority)
recall_majority = recall_score(y_test, y_pred_majority)
f1_majority = f1_score(y_test, y_pred_majority)
confusion_majority = confusion_matrix(y_test, y_pred_majority)

# Display results
print("Majority Voting Accuracy: {:.4f}".format(accuracy_majority))
print("Majority Voting AUPR: {:.4f}".format(aupr_majority))
print("Majority Voting ROC AUC: {:.4f}".format(roc_auc_majority))
print("Majority Voting Precision: {:.4f}".format(precision_majority))
print("Majority Voting Recall: {:.4f}".format(recall_majority))
print("Majority Voting F1 Score: {:.4f}".format(f1_majority))
print("Majority Voting Confusion Matrix:\n", confusion_majority)

# Plot ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_majority)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Majority Voting (AUC = {:.4f})'.format(roc_auc_majority))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Majority Voting ROC Curve')
plt.legend(loc='lower right')
plt.show()