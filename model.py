import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

class Preprocessor(BaseEstimator):

    classifier = svm.svc() # classifieur utilisé

    def __init__(self):
        self.transformer = self #Initialisation des données

    def fit(self, X, y = None): #La méthode fit choisit les hyperparamètres 
        y_train = X['target'].values
        x_train = X.drop('target',axis=1).values 
        pipeline = Pipeline([('pca', PCA()), ('skb', SelectKBest()), ('clf', self.classifier()]) # pipeline permet d’enchaîner le PCA, le SKB et la méthode score() du classifieur vu plus haut
        self.classifier.fit(x_train, y_train)
        t1 = []
        t2 = []
        i = 5
        while (i < 100 ) : #On range dans un tableau les valeurs possible de n_component pour PCA et de k pour la méthode SKB
            t1.append(i)
            i += 5

        for j in range (5, 20) :
            t2.append(j)

        grid_search = GridSearchCV(PCAip, {'pca_n_components':t1, 'SKB_k':t2}, verbose = 1, scoring = make_scorer(accuracy_score()) #grid_search test les différents hyperparamètres et séléctionne les meilleurs
        grid_search.fit(x_train, y_train)
        self.n_pca = grid_search.best_params_.get('pca_n_components')
        self.n_skb = grid_search.best_params_.get('SKB_k')
        return self

    def fit_transform(self, X, y = None): #Appel de la fonction fit() puis transform() dans la même fonction
        return self.fit(X).transform(X)

    def transform(self, X, y = None): #Sélection des données
        y_train = X['target'].values
        x_train = X.drop('target',axis = 1).values 

        vrnc = VarianceThreshold(threshold = (0.05)) # Enlèves les données dont la variance est inférieure à 0.05
        x_train=sel.fit_transform(x_train)

        svd = TruncatedSVD(n_components = self.n_pca) #Réduction du nombre de "features" par données 
        x_train = self.fit_transform(x_train)

        scl = StandardScaler() #Standardisation des données
        x_train = scaler.fit_transform(x_train)

        pca = PCA(n_components = self.n_pca)
        x_train = pca.fit_transform(x_train)
        x_train = kbest.fit_transform(x_train, y_train)
        return x_train



class model (SVC):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.svd = TruncatedSVD(n_components = 10)
        self.classifier= SVC(C=1000.)
        
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        
        
        '''
        self.svd.fit(X)
        X = self.svd.transform(X)
        self.classifier.fit(X,y)
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        # print("Matrix Factorization of test set by SVD")
        # svd = TruncatedSVD(n_components = 100)

        # A = svd.fit_transform(X)
        # T = svd.components_
        
        # print("Shape of A :", A.shape)
        # print("Shape of T :", T.shape)
        X = self.svd.transform(X)
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = self.classifier.predict(X)
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        # y = np.round(X[:,3])
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self