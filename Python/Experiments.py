from SliceLine import SliceLine
from utils import unpickleDataset
from Model import LogisticModel
import math
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder



def SliceLineAdult():
    #get data
    X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/adult/test.pkl")

    encoder = OneHotEncoder()
    encoder.fit(X_train)
    X_train_one_hot = encoder.transform(X_train).toarray()
    X_test_one_hot = encoder.transform(X_test).toarray()
    
    

    #get a list of names for each feature

    with open("./Data/Adult/columnNames.pkl", 'rb') as file:
        featureNames = pkl.load(file)
        
    with open("./Data/Adult/featureMap.pkl", 'rb') as file:
        domains = pkl.load(file)

    #train a logistic regression model and get the per example error
    model = LogisticModel(X_train_one_hot.shape[1])
    model.fit(X_train_one_hot, Y_train, X_test_one_hot, Y_test, 20, .01)
    e = model.per_example_error(X_train_one_hot, Y_train)
    sigma = max(math.ceil(0.01 * X_train.shape[0]), 8)


    #run sliceline
    sf = SliceLine(X_train, e, k=1, sigma=sigma, L=4, auto=True)
    
    print(sf.result)
    SliceLine.pretty_print_results(sf.result, featureNames, domains)
    
SliceLineAdult()