import numpy as np
from Dataset import Dataset, DataLoader
from utils import QuickPlot, getMetrics, unpickleDataset

import numpy.linalg as linalg

from utils import regressionGraph
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder




class LogisticModel:
    def __init__(self, numWeights, VOG=False, VOG_freq=1):
        
        self.W = np.ones(numWeights)
        self.b = 0
        self.metrics = None
        self.VOG = VOG
        self.VOG_freq = VOG_freq
        self.grads = []

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def thresholdedPredict(self, X):
        return (self.predict(X) > 0.5).astype(int)
    
    def compute_loss(self, X, Y):

        preds = self.predict(X)
        
        # avoid log(0)
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        
        # binary cross-entropy loss formula
        loss = -np.mean(Y * np.log(preds) + (1 - Y) * np.log(1 - preds))
        
        return loss
    
    def per_example_error(self, X, Y):
        preds = self.thresholdedPredict(X)
        errors = (preds != Y).astype(int)
        return errors.reshape(-1,1)
    
    
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
        
        
    def fit(self, X_train, Y_train, X_test, Y_test, epochs, lr, decay=0, showMetrics = False):
        
        train = Dataset(X_train, Y_train)
        test = Dataset(X_test, Y_test)


        trainLoader, testLoader = DataLoader(train, batchSize=1000), DataLoader(test, batchSize=1)
        
        for epoch in range(epochs):
            for i, (xs, ys) in trainLoader:
                preds = self.predict(xs)
                errors = preds-ys
                w_grad = np.dot(xs.T, errors) / len(ys)
                b_grad = np.sum(errors) / len(ys)

                self.W -= lr * w_grad
                self.b -= lr * b_grad
            
            if self.VOG and epoch % self.VOG_freq == 0:
                y_hat = self.predict(X_train)
                self.grads.append((y_hat * (1-y_hat)).reshape(-1,1) * np.tile(self.W, (y_hat.shape[0], 1)))
            
            self.metrics = getMetrics(X_train, Y_train, X_test, Y_test, self, self.metrics)
         
        if showMetrics:
            self.showMetrics(epochs)
        
        if self.VOG:
            self.grads = np.stack(self.grads)
            self.VOG = self.computeVOG()

        
        
    def computeVOG(self):
        mean = np.mean(self.grads, axis=0)
        varience = np.var(self.grads, axis=0)
        VOG = np.mean(varience, axis=1)
        return VOG
        
    def showMetrics(self, epochs:int):
        epochs = np.arange(epochs)
        metrics = self.metrics
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





def train_adult():
    X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")

    encoder = OneHotEncoder()
    encoder.fit(X_train)
    X_train = encoder.transform(X_train).toarray()
    X_test = encoder.transform(X_test).toarray()


    model = LogisticModel(128, VOG=False)
    model.fit(X_train, Y_train, X_test, Y_test, 2, .01, showMetrics=False)
    metrics = model.metrics
    #print(metrics)
    
    x1_early = []
    x2_early = []
    
    x1_late = []
    x2_late = []
    
    
    epoch_early = 50
    epoch_late = 199
    
    for i in X_train.T:
        slice_idxs = np.where(i == 1)[0]
        VOGs = VOG[slice_idxs]
        
    for i in metrics["train"].keys():
        if "acc" in i and i != "avg_acc":
            x1_early.append(metrics["train"][i][epoch_early])
            x1_late.append(metrics["train"][i][epoch_late])
            
             
    for i in metrics["train"].keys():
        if "ga" in i:
            x2_early.append(metrics["train"][i][epoch_early])
            x2_late.append(metrics["train"][i][epoch_late])
            
            
                   
    cmap = regressionGraph(x1_early, x2_early, "train early")
    regressionGraph(x1_late, x2_late, "train late", c_list=cmap)
    
    
