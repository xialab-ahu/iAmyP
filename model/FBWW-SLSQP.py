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
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
import matplotlib.pyplot as plt

# Load your existing features and labels
train_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')

y_train = train_dataset['label']
y_train = np.array(y_train)
y_test = test_dataset['label']
y_test = np.array(y_test)

# Load feature datasets
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_morgan_40.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0, delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2, header=0, delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0, delimiter=',')
train_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_AAC.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0, delimiter=',')

test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_morgan_40.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0, delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0, delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0, delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_AAC.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0, delimiter=',')

# Combine features
X_train_data = np.concatenate((train_X_data1, train_X_data2, train_X_data3, train_X_data4), axis=1)
X_test_data = np.concatenate((test_X_data1, test_X_data2, test_X_data3, test_X_data4), axis=1)

# Convert to PyTorch tensors
train_Q = torch.tensor(X_train_data, dtype=torch.float32)
train_K = torch.tensor(X_train_data, dtype=torch.float32)
train_V = torch.tensor(X_train_data, dtype=torch.float32)
test_Q = torch.tensor(X_test_data, dtype=torch.float32)
test_K = torch.tensor(X_test_data, dtype=torch.float32)
test_V = torch.tensor(X_test_data, dtype=torch.float32)

# Define Attention mechanism
class Attention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(0, 1))
        attention_weights = torch.softmax(scores, dim=1)
        output = torch.matmul(attention_weights, V)
        return output

# Initialize Attention model
attention_model = Attention()

# Apply attention mechanism
X_train = attention_model(train_Q, train_K, train_V)
X_test = attention_model(test_Q, test_K, test_V)

# Define base classifiers
base_classifiers = [
    SVC(kernel='linear', gamma=1.0, C=0.163, probability=True),
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191, max_depth=3, min_samples_leaf=3, min_samples_split=20, random_state=0),
    RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0),
    DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
    ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0),
    lgb.LGBMClassifier(n_estimators=129, learning_rate=0.04392776161174538, max_depth=3, min_child_samples=20, num_leaves=35, random_state=0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=1000, learning_rate=0.8),
    KNeighborsClassifier(n_neighbors=2),
    LogisticRegression(penalty='l2', solver='lbfgs', C=0.0012086328436524494, max_iter=1000)
]

# Perform feature bagging for each base classifier
train_meta_features = np.zeros((X_train.shape[0], len(base_classifiers)))
test_meta_features = np.zeros((X_test.shape[0], len(base_classifiers)))

for i, clf in enumerate(base_classifiers):
    selected_features = np.random.choice(X_train.shape[1], size=int(0.8 * X_train.shape[1]), replace=False)
    X_train_bagged = X_train[:, selected_features]
    X_test_bagged = X_test[:, selected_features]

    clf.fit(X_train_bagged, y_train)
    train_meta_features[:, i] = clf.predict_proba(X_train_bagged)[:, 1]
    test_meta_features[:, i] = clf.predict_proba(X_test_bagged)[:, 1]

# Train the meta-classifier
from scipy.optimize import minimize

# Define function to minimize (log loss)
def log_loss_func(weights):
    final_prediction = np.zeros_like(test_meta_features[:, 0])
    for weight, prediction in zip(weights, test_meta_features.T):
        final_prediction += weight * prediction
    return log_loss(y_test, final_prediction)

# Initial weights and constraints
starting_values = [1.0 / len(base_classifiers)] * len(base_classifiers)
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(base_classifiers)

# Perform optimization to find optimal weights
res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

# Print results
print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

# Ensemble predictions using optimal weights
y_prob_ensemble = np.dot(test_meta_features, res['x'])

# Evaluate performance metrics
fpr, tpr, thresholds = roc_curve(y_test, y_prob_ensemble)
roc_auc = auc(fpr, tpr)

precision = precision_score(y_test, (y_prob_ensemble > 0.5).astype(int))
recall = recall_score(y_test, (y_prob_ensemble > 0.5).astype(int))
f1 = f1_score(y_test, (y_prob_ensemble > 0.5).astype(int))
confusion = confusion_matrix(y_test, (y_prob_ensemble > 0.5).astype(int))
accuracy = accuracy_score(y_test, (y_prob_ensemble > 0.5).astype(int))
aupr = average_precision_score(y_test, y_prob_ensemble)
mcc = matthews_corrcoef(y_test, (y_prob_ensemble > 0.5).astype(int))

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('ROC_curve.png')
plt.show()

# Plot Confusion Matrix
plt.figure()
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUPR:", aupr)
print("MCC:", mcc)
