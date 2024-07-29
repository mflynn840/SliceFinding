import numpy as np
from Dataset import Dataset, DataLoader
from utils import QuickPlot, prepAdult
from sklearn.metrics import accuracy_score
import numpy.linalg as linalg

class LogisticModel:
    def __init__(self, numWeights):
        
        self.W = np.ones(numWeights)
        self.b = 0
        
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
        
        
        accs = []
        
        for epoch in range(epochs):
            for i, (xs, ys) in trainLoader:
                preds = self.predict(xs)
                errors = preds-ys
                w_grad = np.dot(xs.T, errors) / len(ys)
                b_grad = np.sum(errors) / len(ys)


                self.W -= lr * w_grad
                self.b -= lr * b_grad
            
            
            X_train, Y_train = train[:]
            preds = self.predict(X_train)
            error = preds - Y_train
            
            preds = self.thresholdedPredict(X_train)

            
            acc = accuracy_score(Y_train, preds)
            accs.append(acc)
            
        epochs = np.arange(epochs)
        QuickPlot([epochs], [accs], ["SGD LogReg"], "Epoch", "accuracy", "Logistic Regression on Adult")


            

X_train, Y_train, X_test, Y_test = prepAdult()

foo = LogisticModel(86)
foo.trainAdult(100, .01)

print(foo.GAError(X_train, Y_train, X_train, Y_train))

    
    