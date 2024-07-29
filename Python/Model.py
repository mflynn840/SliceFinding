import numpy as np
from Dataset import Dataset, DataLoader
from utils import QuickPlot, prepAdult, getMetrics
from sklearn.metrics import accuracy_score
import numpy.linalg as linalg
import pickle as pkl


class LogisticModel:
    def __init__(self, numWeights):
        
        self.W = np.ones(numWeights)
        self.b = 0
        self.metrics = {
                    "train": {
                        "bm_acc": [],
                        "bf_acc": [],
                        "wm_acc": [],
                        "wf_acc": [],
                        "avg_acc" : []
                    },
                    "test": {
                        "bm_acc": [],
                        "bf_acc": [],
                        "wm_acc": [],
                        "wf_acc": [],
                        "avg_acc" : []
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
        s2_grad = np.dot(X2.T, errors1) / len(X2)
        
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
        
        metrics_list = [
            metrics["test"]["bm_acc"],
            metrics["test"]["bf_acc"],
            metrics["test"]["wm_acc"],
            metrics["test"]["wf_acc"],
            metrics["test"]["avg_acc"]
        ]   
        
        QuickPlot([epochs, epochs, epochs, epochs, epochs, epochs], metrics_list, ["black male (n=726)", "black female (n=685)", "white male (n=8978)", "white female (n=3987)", "average"], "Epoch", "Group Accuracy", "Test Set Accuracy on Adult")
            

X_train, Y_train, X_test, Y_test = prepAdult()

foo = LogisticModel(86)
foo.trainAdult(200, .01)

print(foo.GAError(X_train, Y_train, X_train, Y_train))

    
    