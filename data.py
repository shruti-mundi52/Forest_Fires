# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:26:52 2020

@author: Lenovo
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA

def dataset(file_name=""):
    data=pd.read_csv(file_name)
    df=pd.DataFrame(data)
    return df

def scaling(data=''):
    scale=StandardScaler()
    data=scale.fit_transform(data)
    return data
    
def pca_apply(data,no_of_components):
    pca=PCA(n_components=no_of_components)
    x_new=pca.fit_transform(data)
    explained_variance=pca.explained_variance_ratio_
    components=pca.components_
    return x_new,explained_variance,components