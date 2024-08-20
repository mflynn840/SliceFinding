from SliceLine import SliceFinder
from utils import unpickleDataset
from Model import LogisticModel
import math
import pickle as pkl

#get data
X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/adult/test.pkl")

#get a list of names for each feature

with open("./Data/Adult/columnNames.pkl", 'rb') as file:
    featureNames = pkl.load(file)

#train a logistic regression model and get the per example error
model = LogisticModel(86)
model.fit(X_train, Y_train, X_test, Y_test, 20, .01)
e = model.per_example_error(X_train, Y_train)
sigma = max(math.ceil(0.01 * X_train.shape[0]), 8)


#run sliceline
sf = SliceFinder(X_train, e, k=3, sigma=sigma, L=1, auto=True)
SliceFinder.pretty_print_results(sf.result, featureNames)