import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

accuracy_pca, roc_auc_pca, accuracy_cls, roc_auc_cls, ar_cls, cm_cls = [], [], [], [], [], []


class Optimization():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.q = 5
        self.d = 5
        self.q_nbr = 6
        self.q_min = 3
        self.q_opt = 3
        self.opt_n = 2
        self.classifier_kfold = None
        self.X_train_kfold = None

                
        
    def pca_optimization(self):
        for n in range(1,self.d):
            pca = PCA(n_components = n)
            X_train_pca = pca.fit_transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)
            explained_variance = pca.explained_variance_ratio_
        
            # COMMAND BELOW SHOULD LIST VARIANCES OF EACH VARIABLE USED TO CREATE PRINCIPLE COMPONENTS
            # print( pd.DataFrame(pca.components_,index = ['PC-1','PC-2']))
            
            # Fitting Logistic Regression to the Training set
            classifier = KNeighborsClassifier(n_neighbors = self.q, metric = "minkowski", p = 2)
            classifier.fit(X_train_pca, self.y_train)
            
            # Predicting the Test set results
            y_pred = classifier.predict(X_test_pca)
            
            predictions = [round(value) for value in y_pred]
            
            # Evaluate predictions
            accuracy = accuracy_score(self.y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            
            # Area Under the Receiver Operating Characteristic Curve
            roc_auc = roc_auc_score(self.y_test, predictions)
            print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)
            
            # Making the Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            accuracy_pca.append(accuracy)
            roc_auc_pca.append(roc_auc)
            print(n)
        
            b = 0
            if n == self.d-1:
                for i in range(1,self.d-1):
                    if accuracy_pca[i] > accuracy_pca[b]:
                        b = i
        
        # Set optimal number of PCA components            
        self.opt_n = b+1
                    
        print("-------------------------------------------")
        print("The optimal number of PCA components is: ", b+1)
        print("-------------------------------------------")
            
        return accuracy_pca, roc_auc_pca
    
    def cls_optimization(self):
        while self.q_opt < self.q_nbr:
            pca = PCA(n_components = self.opt_n)
            X_train_cls = pca.fit_transform(self.X_train)
            X_test_cls = pca.transform(self.X_test)
            explained_variance = pca.explained_variance_ratio_
            
            # Fitting Logistic Regression to the Training set
            classifier = KNeighborsClassifier(n_neighbors = self.q_opt, metric = "minkowski", p = 2)
            classifier.fit(X_train_cls, self.y_train)
            
            # Predicting the Test set results
            y_pred = classifier.predict(X_test_cls)
            
            predictions = [round(value) for value in y_pred]
            
            print("--------------------------------------")
            print("Using", self.q_opt, "Kmeans neighbors and", self.opt_n, "PCA components: ")
            
            # evaluate predictions
            accuracy = accuracy_score(self.y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            
            # Area Under the Receiver Operating Characteristic Curve
            roc_auc = roc_auc_score(self.y_test, predictions)
            print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)
            
            # Making the Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
                    
            accuracy_cls.append(accuracy)
            roc_auc_cls.append(roc_auc)
            ar = [[self.q_opt],[self.opt_n],[accuracy],[roc_auc]]
            ar_cls.append(ar)
            cm_cls.append(cm)
            
            self.q_opt += 1
            
        self.classifier_kfold = classifier
        self.X_train_kfold = X_train_cls
    
        return ar_cls, cm_cls

    def kfold_optimization(self):
        # Applying k-Fold Cross Validation (GET 10 DIFFERENT ACCURACYS AND AVERAGE THEM)
        accuracies = cross_val_score(estimator = self.classifier_kfold, X = self.X_train_kfold, y = self.y_train, cv = 10)
        print("Accuracy Average:", accuracies.mean())
        print("Accuracy Standard Deviation:", accuracies.std())
        return accuracies.mean()


    def build_classifier(self, optimizer='adam'):
        #Build the neural network
        dim = int(np.size(self.X_train, 1))
        classifier = Sequential()
        classifier.add(Dense(output_dim = int(dim/2), init = 'uniform', activation = 'relu', input_dim = dim))
        classifier.add(Dense(output_dim = int(dim/2/2), init = 'uniform', activation = 'relu'))
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    
    def keras_optimization(self, batch_min, batch_max, epoch_min, epoch_max):
        # Will be trained with KFold (n_jobs - # of CPUs to use [-1 uses all of them])
        classifier = KerasClassifier(build_fn = self.build_classifier)
        parameters = {'batch_size': [batch_min, batch_max], 'nb_epoch': [epoch_min, epoch_max], 'optimizer': ['adam', 'rmsprop']}
        grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
        
        grid_search = grid_search.fit(self.X_train, self.y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        
        print("Best Parameters:", best_parameters)
        print("Best Accuracy:", best_accuracy)
        
        return best_parameters, best_accuracy
    

