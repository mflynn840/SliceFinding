import numpy as np
from Dataset import Dataset, DataLoader
from utils import QuickPlot, prepAdult, getMetrics
from sklearn.metrics import accuracy_score
import numpy.linalg as linalg
import pickle as pkl
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from utils import regressionGraph

class LogisticModel:
    def __init__(self, numWeights):
        
        self.W = np.ones(numWeights)
        self.b = 0
        self.metrics = {
                    "train": {
                        "bm_acc": [],  # Black male accuracy
                        "bf_acc": [],  # Black female accuracy
                        "wm_acc": [],  # White male accuracy
                        "wf_acc": [],  # White female accuracy
                        "bm_ga": [],   # Black male GAError
                        "bf_ga": [],   # Black female GAError
                        "wm_ga": [],   # White male GAError
                        "wf_ga": [],   # White female GAError
                        "avg_acc": []  # Average accuracy across all groups
                    },
                    "test": {
                        "bm_acc": [],  # Black male accuracy
                        "bf_acc": [],  # Black female accuracy
                        "wm_acc": [],  # White male accuracy
                        "wf_acc": [],  # White female accuracy
                        "bm_ga": [],   # Black male GAError
                        "bf_ga": [],   # Black female GAError
                        "wm_ga": [],   # White male GAError
                        "wf_ga": [],   # White female GAError
                        "avg_acc": []  # Average accuracy across all groups
                    }
                }
        
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def thresholdedPredict(self, X):
        return (self.predict(X) > 0.5).astype(int)
    
    
    #x1/y1 is a superset of X2/y2
    def GAError(self, X1, Y1, X2, Y2):
        preds1 = self.predict(X1)
        preds2 = self.predict(X2)
        
        errors1 = preds1-Y1
        errors2 = preds2-Y2
        
        s1_grad = np.dot(X1.T, errors1) / len(X1)
        s2_grad = np.dot(X2.T, errors2) / len(X2)
        
        normGradDiff = linalg.norm(s1_grad - s2_grad)
        return normGradDiff
        
        
    def trainAdult(self, epochs, lr, decay=0):
        train, test = Dataset.loadAdult()
        trainLoader, testLoader = DataLoader(train, batchSize=1000), DataLoader(test, batchSize=1)
        
        X_train, Y_train = train[:]
        X_test, Y_test = test[:]

        
        for epoch in range(epochs):
            for i, (xs, ys) in trainLoader:
                preds = self.predict(xs)
                errors = preds-ys
                w_grad = np.dot(xs.T, errors) / len(ys)
                b_grad = np.sum(errors) / len(ys)

                self.W -= lr * w_grad
                self.b -= lr * b_grad
            
            metrics = getMetrics(X_train, Y_train, X_test, Y_test, self, self.metrics)
         
        epochs = np.arange(epochs)
        metrics = getMetrics(X_train, Y_train, X_test, Y_test, self, self.metrics, last=True)
        test_ga_list = [
            metrics["test"]["bm_ga"],
            metrics["test"]["bf_ga"],
            metrics["test"]["wm_ga"],
            metrics["test"]["wf_ga"],
        ]   
        
        train_ga_list = [
            metrics["train"]["bm_ga"],
            metrics["train"]["bf_ga"],
            metrics["train"]["wm_ga"],
            metrics["train"]["wf_ga"],
        ]   
        
        test_acc_list = [
            metrics["test"]["bm_acc"],
            metrics["test"]["bf_acc"],
            metrics["test"]["wm_acc"],
            metrics["test"]["wf_acc"],
            metrics["test"]["avg_acc"]
        ]
        
        train_acc_list = [
            metrics["train"]["bm_acc"],
            metrics["train"]["bf_acc"],
            metrics["train"]["wm_acc"],
            metrics["train"]["wf_acc"],
            metrics["train"]["avg_acc"]
        ]
        
        #test set GA error vs. epoch
        #QuickPlot([epochs, epochs, epochs, epochs, epochs], test_ga_list, ["black male (n=726)", "black female (n=685)", "white male (n=8978)", "white female (n=3987)"], "Epoch", "Group GA error", "Test set GA error")
        
        #test acc vs. epoch
        QuickPlot([epochs, epochs, epochs, epochs, epochs, epochs], test_acc_list, ["black male (n=726)", "black female (n=685)", "white male (n=8978)", "white female (n=3987)", "average"], "Epoch", "Accuracy", "Test set accuracy", markLast=True, percent=True)
         
        #train set GA error vs. epoch 
        #QuickPlot([epochs, epochs, epochs, epochs, epochs], train_ga_list, ["black male (n=726)", "black female (n=685)", "white male (n=8978)", "white female (n=3987)"], "Epoch", "Group GA error", "Train set GA error")
         
        #train set accuracy vs. epoch
        QuickPlot([epochs, epochs, epochs, epochs, epochs, epochs], train_acc_list, ["black male (n=726)", "black female (n=685)", "white male (n=8978)", "white female (n=3987)", "average"], "Epoch", "Accuracy", "Train set Accuracy")
        
        #train regressions
        #regressionGraph(metrics["train"]["bm_acc"], metrics["train"]["bm_ga"], "Black Male", "train")
        #regressionGraph(metrics["train"]["bf_acc"], metrics["train"]["bf_ga"], "Black Female", "train")
        #regressionGraph(metrics["train"]["wm_acc"], metrics["train"]["wm_ga"], "White Male", "train")
        #regressionGraph(metrics["train"]["wf_acc"], metrics["train"]["wf_ga"], "White Female", "train")

        # Test set - Black Male
        #regressionGraph(metrics["test"]["bm_acc"], metrics["test"]["bm_ga"], "Black Male", "test")
        #regressionGraph(metrics["test"]["bf_acc"], metrics["test"]["bf_ga"], "Black Female", "test")
        #regressionGraph(metrics["test"]["wm_acc"], metrics["test"]["wm_ga"], "White Male", "test")
        #regressionGraph(metrics["test"]["wf_acc"], metrics["test"]["wf_ga"], "White Female", "test")

X_train, Y_train, X_test, Y_test = prepAdult()

foo = LogisticModel(86)
foo.trainAdult(200, .01)

print(foo.GAError(X_train, Y_train, X_train, Y_train))

    
    