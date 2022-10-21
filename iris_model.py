import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_model():
    # Load the data
    iris=pd.read_csv("iris.csv", sep=",")
    X=np.array(iris[['sepal_length','sepal_width','petal_length','petal_width']].values.tolist())
    Y=np.array(iris['species'].values.tolist()) 
    for index in range(0,len(Y)):
            if Y[index] == 'setosa':
                Y[index] = 0
            elif Y[index] == 'versicolor':
                Y[index] = 1
            elif Y[index] == 'virginica':
                Y[index] = 2
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)  
    testSet = np.append(X_test, y_test.reshape(30,1), axis=1)
    # Save the test set for future use
    pd.DataFrame(testSet).to_csv('test_set.csv') 

    # train and cross validation
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    import warnings
    warnings.filterwarnings("ignore")   
    params = {'C':[0.01, 0.1, 1, 10, 100, 1000],
              'max_iter':[10, 30, 100, 300],
              'class_weight':['balanced', None],
              'solver':['liblinear','sag','lbfgs','newton-cg']
             }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid=params, cv=4)
    clf.fit(X_train, y_train)
    print("The chosen parameters are:",clf.best_params_)
    # Run the test set 
    accuracy = clf.score(X_test, y_test)
    print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy))
    # Save the model
    from sklearn.externals import joblib
    joblib.dump(clf, 'iris_model.model')
    print("The model is saved as iris_model.model")
    
if __name__ == '__main__':
    train_model()