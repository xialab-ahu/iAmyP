import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import shap

train_dataset = pd.read_csv('D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv('D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')
y_train = train_dataset['label']
y_train = np.array(y_train)
y_test = test_dataset['label']
y_test = np.array(y_test)
y = np.concatenate((y_train,y_test), axis=0)
train_dataset_feature_name = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\train_data_feature.csv'
train_dataset_feature= pd.read_csv(train_dataset_feature_name, header=0,  delimiter=',')
test_dataset_feature_name = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\test_data_feature.csv'
test_dataset_feature = pd.read_csv(test_dataset_feature_name, header=0,  delimiter=',')
X_train = np.array(train_dataset_feature)
X_test = np.array(test_dataset_feature)

print(X_train.shape,X_test.shape)
clf_xgb = xgb.XGBClassifier(n_estimators=100, random_state=0)
clf_xgb.fit(X_train, y_train)

explainer = shap.TreeExplainer(clf_xgb)
shap_values = explainer.shap_values(X_test)
feature_names = [f'MF_{i}' for i in range(1, 1025)] + [f'DDE_{i}' for i in range(1, 401)] + [f'blousum_{i}' for i in range(1, 121)]+[f'AAC_{i}' for i in range(1, 21)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

shap.summary_plot(shap_values, X_test, max_display=30, show=False, title="SHAP Summary Plot", feature_names=feature_names, plot_size=(12, 10))
# Set x-axis labels at the center of each column and adjust font size

plt.sca(ax1)
shap.summary_plot(shap_values, X_test,max_display=30,  plot_type="bar", show=False, title="SHAP Summary Plot (Bar)", feature_names=feature_names, plot_size=(12, 10))
ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_yticklabels([])
plt.sca(ax2)

fig.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.xticks( fontsize=14)
plt.yticks(fontsize=14)
plt.show()







