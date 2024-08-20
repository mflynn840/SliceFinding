from SliceLine import SliceFinder
from utils import unpickleDataset
from Model import LogisticModel

#get data
X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/adult/test.pkl")

#get a list of names for each feature
featureNames = ["feature" + str(i) for i in range(86)]

#train a logistic regression model and get the per example error
model = LogisticModel(86)
model.fit(X_train, Y_train, X_test, Y_test, 20, .01)
e = model.per_example_error(X_train, Y_train)

#run sliceline
sf = SliceFinder(X_train, e, auto=True)
SliceFinder.pretty_print_results(sf.result, featureNames)