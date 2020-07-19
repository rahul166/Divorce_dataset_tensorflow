import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
##Data preprocessing part
data=pd.read_csv("divorce.csv")
# print(data.head())
sc=StandardScaler()
features=data.iloc[:,:54]
features=features.to_numpy()
final_features=sc.fit_transform(features)
# print(final_features)
labels=data.iloc[:,54:]
labels=labels.replace(1,'yes')
labels=labels.replace(0,'No')
# print(labels)
# print(pd.get_dummies(labels))
final_labels=pd.get_dummies(labels)
final_labels=final_labels.to_numpy()
print(final_labels)



with open('data','wb') as f:
	pickle.dump([final_features,final_labels],f)



