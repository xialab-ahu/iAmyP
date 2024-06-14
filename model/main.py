import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
import torch.optim as optim
from scipy.optimize import minimize
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import seaborn as sns
import shap
from scipy.stats import gaussian_kde
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Load your existing features and labels
# Load your existing features and labels
train_dataset = pd.read_csv(r'/Hexapeptide Amyloidogenesis/data/TrainingDataset.csv')
test_dataset = pd.read_csv(r'/Hexapeptide Amyloidogenesis/data/TestingDataset.csv')

y_train = train_dataset['label']
y_train = np.array(y_train)
y_test = test_dataset['label']
y_test = np.array(y_test)
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_morgan_40.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0,  delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2,header=0,  delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0, delimiter=',')
train_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_AAC.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0,  delimiter=',')
X_train_data = np.concatenate((train_X_data1 ,train_X_data2,train_X_data3,train_X_data4), axis=1)
test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_morgan_40.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0,  delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0,  delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0,delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_AAC.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0,delimiter=',')
X_test_data = np.concatenate((test_X_data1,test_X_data2,test_X_data3,test_X_data4), axis=1)


train_Q = torch.tensor(X_train_data, dtype=torch.float32)
train_K = torch.tensor(X_train_data, dtype=torch.float32)
train_V = torch.tensor(X_train_data, dtype=torch.float32)
test_Q = torch.tensor(X_test_data , dtype=torch.float32)
test_K = torch.tensor(X_test_data , dtype=torch.float32)
test_V = torch.tensor(X_test_data , dtype=torch.float32)

# 定义注意力模块
class Attention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(0, 1))
        attention_weights = torch.softmax(scores, dim=1)
        output = torch.matmul(attention_weights, V)
        return output


attention_model = Attention()

X_train = attention_model(train_Q, train_K, train_V)
X_test = attention_model(test_Q, test_K, test_V)

# Define classifiers
clf_svm = SVC(kernel='linear', gamma=1.0,C=0.163, probability=True)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

clf_gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.04374442322305191, max_depth=3,min_samples_leaf=3, min_samples_split=20,random_state=0)
clf_gbdt.fit(X_train, y_train)
y_pred_gbdt = clf_gbdt.predict(X_test)

clf_rf = RandomForestClassifier(n_estimators=375, max_depth=10, min_samples_split=2,min_samples_leaf=1, random_state=0)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

clf_ert = ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0)
clf_ert.fit(X_train, y_train)
y_pred_ert = clf_ert.predict(X_test)

# 请根据你的LightGBM模型设置相应的参数
clf_lightgbm = lgb.LGBMClassifier(n_estimators=129,learning_rate=0.04392776161174538,max_depth=3,min_child_samples=20,num_leaves=35, random_state=0)
clf_lightgbm.fit(X_train, y_train)
y_pred_lightgbm = clf_lightgbm.predict(X_test)

clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=1000, learning_rate=0.8)
clf_ada.fit(X_train, y_train)
y_pred_ada = clf_ada.predict(X_test)
# 训练和预测kNN模型
clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

# 训练和预测逻辑回归模型
clf_lr = LogisticRegression(penalty='l2',solver='lbfgs',C=0.0012086328436524494,max_iter=1000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)


#  构建模型预测矩阵
predictions_nb = np.vstack((y_pred_svm,y_pred_gbdt,y_pred_rf, y_pred_dt , y_pred_ert ,y_pred_lightgbm,y_pred_knn,y_pred_ada,y_pred_lr))

log_loss_history = []
def log_loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_nb):
        final_prediction += weight * prediction
    # return final_prediction
    return metrics.log_loss(y_test, final_prediction)
# 初始权重和约束
starting_values = [1 / len(predictions_nb)] * len(predictions_nb)
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(predictions_nb)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
matrix_prod = np.dot(res['x'], predictions_nb)
y_prob_nb = matrix_prod
# # 将y_prob_nb转换为数据框
df = pd.DataFrame(y_prob_nb, columns=['Probability'])
# 将数据框保存到CSV文件
df.to_csv('y_prob_nb_output.csv', index=False)
#
# # 计算ROC曲线
fpr, tpr, thresholds2 = metrics.roc_curve(y_test, y_prob_nb)
print("fpr: ", fpr)
roc_auc = metrics.auc(fpr, tpr)

# 计算其他性能指标
y_true = y_test
y_pred_ensemble = [np.float(each > 0.5) for each in y_prob_nb]

accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)
print("ACC",accuracy_score_ensemble )
# # 绘制ROC曲线
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# 计算AUPR
aupr = average_precision_score(y_true, y_prob_nb)
# 输出AUPR值
print("AUPR: ", aupr)
# 计算精确度
precision = precision_score(y_true, y_pred_ensemble)

# 计算召回率
recall = recall_score(y_true, y_pred_ensemble)

# 计算F1分数
f1 = f1_score(y_true, y_pred_ensemble)

# 计算混淆矩阵
confusion = confusion_matrix(y_true, y_pred_ensemble)
# Define class labels
class_names = ['0', '1']

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.set()
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
# plt.title('REF-Attention-Ensemble Confusion Matrix')
plt.show()

print("精确度: ", precision)
print("召回率: ", recall)
print("F1分数: ", f1)
print("混淆矩阵: ")
print(confusion)
# 计算集成模型的MCC值
mcc_ensemble = matthews_corrcoef(y_true, y_pred_ensemble)
print(f"Ensemble MCC: {mcc_ensemble}")
