import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn import metrics
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

train_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')
y_train = np.array(train_dataset['label'])
y_test = np.array(test_dataset['label'])
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_morgan_40.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0, delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2, header=0, delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0, delimiter=',')
train_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_AAC.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0, delimiter=',')
X_train = np.concatenate((train_X_data1, train_X_data2, train_X_data3, train_X_data4), axis=1)
test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_morgan_40.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0, delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0, delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0, delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_AAC.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0, delimiter=',')
X_test = np.concatenate((test_X_data1, test_X_data2, test_X_data3, test_X_data4), axis=1)
train_Q = torch.tensor(X_train, dtype=torch.float32)
train_K = torch.tensor(X_train, dtype=torch.float32)
train_V = torch.tensor(X_train, dtype=torch.float32)
test_Q = torch.tensor(X_test, dtype=torch.float32)
test_K = torch.tensor(X_test, dtype=torch.float32)
test_V = torch.tensor(X_test, dtype=torch.float32)

class Attention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(0, 1))
        attention_weights = torch.softmax(scores, dim=1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

attention_model = Attention()
X_train, attention_weights_train = attention_model(train_Q, train_K, train_V)
X_test, attention_weights_test = attention_model(test_Q, test_K, test_V)

# Initialize base classifiers
base_classifiers = [
    # SVC(kernel='linear', gamma=1.0, C=0.163, probability=True),
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191, max_depth=3, min_samples_leaf=3,
                               min_samples_split=20, random_state=0),
    RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0),
    DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
    ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0),
    lgb.LGBMClassifier(n_estimators=129, learning_rate=0.04392776161174538, max_depth=3, min_child_samples=20,
                       num_leaves=35, random_state=0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                       algorithm="SAMME",
                       n_estimators=1000, learning_rate=0.8),
    KNeighborsClassifier(n_neighbors=2),
    LogisticRegression(penalty='l2', solver='lbfgs', C=0.0012086328436524494, max_iter=1000),
    xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
]

# Create an empty array to store out-of-fold predictions
train_meta_features = np.zeros((X_train.shape[0], len(base_classifiers)))
test_meta_features = np.zeros((X_test.shape[0], len(base_classifiers)))

kf = KFold(n_splits=5, shuffle=True, random_state=0)

for i, clf in enumerate(base_classifiers):
    test_meta_features_fold = np.zeros((X_test.shape[0], kf.get_n_splits()))
    for j, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        clf.fit(X_fold_train, y_fold_train)
        train_meta_features[val_index, i] = clf.predict_proba(X_fold_val)[:, 1]
        test_meta_features_fold[:, j] = clf.predict_proba(X_test)[:, 1]
    test_meta_features[:, i] = test_meta_features_fold.mean(axis=1)

# Train the meta-classifier
meta_classifier = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
meta_classifier.fit(train_meta_features, y_train)
y_prob_final = meta_classifier.predict_proba(test_meta_features)[:, 1]

fpr, tpr, thresholds2 = metrics.roc_curve(y_test, y_prob_final)
roc_auc = metrics.auc(fpr, tpr)
y_pred_ensemble = [float(each > 0.5) for each in y_prob_final]
accuracy_score_ensemble = metrics.accuracy_score(y_test, y_pred_ensemble)
print("ACC", accuracy_score_ensemble)
labels = ['AUC = %0.3f' % roc_auc]
plt.figure()
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.plot(fpr, tpr, label=labels)
plt.legend(loc='lower right')
plt.savefig(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\AUC_compared.png')
plt.show()

aupr = average_precision_score(y_test, y_prob_final)
print("AUPR: ", aupr)
precision = precision_score(y_test, y_pred_ensemble)
recall = recall_score(y_test, y_pred_ensemble)
f1 = f1_score(y_test, y_pred_ensemble)
confusion = confusion_matrix(y_test, y_pred_ensemble)
class_names = ['0', '1']
plt.figure(figsize=(8, 6))
sns.set()
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('REF-Attention-Ensemble Confusion Matrix')
plt.show()

print("精确度: ", precision)
print("召回率: ", recall)
print("F1分数: ", f1)
print("混淆矩阵: ")
print(confusion)
mcc_ensemble = matthews_corrcoef(y_test, y_pred_ensemble)
print(f"Ensemble MCC: {mcc_ensemble}")
