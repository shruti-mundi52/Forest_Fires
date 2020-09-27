# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:19:24 2020

@author: Lenovo
"""


import data

import pandas as pd
import numpy as np

#importing the dataset
df=dataset('forestfires.csv')
  

x=df.iloc[:,4:13]
#scaling the dataest
scaled_x=scaling(x)


#Label encoding categorical data
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df.iloc[:,2]=lb.fit_transform(df.iloc[:,2])
df.iloc[:,3]=lb.fit_transform(df.iloc[:,3])




#Applying PCA
x_new,explained_variance,components=pca_apply(scaled_x,4)

features=['FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
scaled_x=pd.DataFrame(scaled_x,columns=features)

#Dataframe consisting of components 
df_c=pd.DataFrame(components,columns=scaled_x.columns,index=['pc-1','pc-2','pc-3','pc-4'])


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=2, random_state=0)
k_means.fit(x_new)


clusters = k_means.labels_.tolist()

#DataFrame containing the data along with the clusters(LABELLED DATA)
frame=pd.DataFrame(clusters,columns=['clusters'])
df_new=df.drop(labels=['X','Y','day','rain','RH','wind','area'],axis=1)
frame_new=pd.concat([df_new,frame],axis=1)

#dataframe containing the reduced data along with the clusters
x_new_df=pd.DataFrame(x_new)
clusters=pd.DataFrame(clusters)
frame_demo=pd.concat([x_new_df,clusters],axis=1)

frame_new.to_csv('forest_labelled.csv')

#plotting the data
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
scatter=ax.scatter(frame_new.iloc[:,4],frame_new.iloc[:,7],c=frame_new['clusters'],s=50)
ax.legend(*scatter.legend_elements())
ax.set_title('K Means Clustering')
ax.set_xlabel('FFMC')
ax.set_ylabel('ISI')
plt.colorbar(scatter)





#Classification
x_class=frame_new.iloc[:,:-1]
y_class=frame_new.iloc[:,13]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_class,y_class,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)

y_pred=lg.predict(x_test)



from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred_dt=dt.predict(x_test)

accuracy_score(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_dt)
print(classification_report(y_test,y_pred_dt))
