from sys import argv
import warnings
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC

import numpy as np

class Preprocessor(BaseEstimator):

    def __init__(self):
        self.transformer = self #Initialisation des données
        self.transformer.classifier = SVC()#classifieur choisi

    def fit(self, X, y = None ): 
         return self.transformer

    def fit_transform(self, X, y = None): #Appel de la fonction fit() puis transform() dans la même fonction
        return self.transformer.fit(X).transform(X)

    def transform(self, X, y = None ):  
        vrnc = VarianceThreshold(threshold = (0.05)) # Enlèves les données dont la variance est inférieure à 0.05
        svd = TruncatedSVD(n_components = 12) #Choix du nombres de features max pour le pca
        pca = PCA(n_components = 10)
        scl = StandardScaler() #Fonction qui standardise les données 
        X = scl.fit_transform(X)#Standardisation des données
        X = pca.fit_transform(X)#Réduction du nombre de features par données
        return X
