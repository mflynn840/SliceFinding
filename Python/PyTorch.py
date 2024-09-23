import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from utils import QuickPlot, get_slice_idxs
import torch.profiler
from math import sqrt, floor
import torch.nn.functional as F


class RunningVOG:
    def __init__(self, shape):
        self.shape = shape
        self.n = 0
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)
    
    def update(self, data):
        self.n += 1
        delta = data-self.mean
        self.mean += delta/self.n
        delta2 = data-self.mean
        self.m2 += delta*delta2
    
    def get_mean(self):
        return self.mean

    def get_varience(self):
        if self.n >1:
            return self.m2 / (self.n-1)
    
    def get_VOGs(self):
        varience = self.get_varience()
        VOG = (1/self.shape[1]) * torch.sum(varience, 1)
        return VOG
    
    def get_slice_VOGs(self, slice_idxs):
        all_vogs = self.get_VOGs()
        
        slice_VOGS = []
        for i in range(len(slice_idxs)):
            slice_VOGS.append(torch.mean(all_vogs[slice_idxs[i]]))
            
        return torch.stack(slice_VOGS)
            

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

class ConvexNN(nn.Module):
    def __init__(self, inFeatures, outClasses):
        super(ConvexNN, self).__init__()
        self.linear = nn.Linear(inFeatures, outClasses)
    
    def forward(self, x):
        return self.linear(x)
    
    
class SimpleNN(nn.Module):
    
    def __init__(self, inFeatures, hiddenNeurons, classesOut):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(inFeatures, hiddenNeurons)
        self.hidden2 = nn.Linear(hiddenNeurons, hiddenNeurons)
        self.output = nn.Linear(hiddenNeurons,classesOut)
        self.flatten = nn.Flatten()
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x
    
    def pre_softmax_activations(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        return x
    
    def per_slice_loss(model, X, Y, slice_idx_list):
        
        X = X.to("cuda")
        Y = Y.to("cuda")
        
        loss_fn = nn.CrossEntropyLoss()
        
        slice_losses = []
        with torch.no_grad():
            for i in range(len(slice_idx_list)):
                slice_idxs = slice_idx_list[i]
                logits = model(X[slice_idxs])
                slice_losses.append(loss_fn(logits, Y[slice_idxs]).item())
        return np.asarray(slice_losses)
    
    def per_slice_accuracy(model, X, Y, slice_idx_list):
        
        X = X.to("cuda")
        Y = Y.to("cuda")
        
        slice_accs = []
        with torch.no_grad():
            for i in range(len(slice_idx_list)):
                slice_idxs = slice_idx_list[i]
                logits = model(X[slice_idxs])
                preds = torch.argmax(logits, 1)
                slice_accs.append(((preds == Y[slice_idxs]).sum().item())/slice_idxs.shape[0])
        
        return np.asarray(slice_accs)
        
    
    def get_slice_ga_errors(model, X, Y, slice_idxs):
        device = "cuda:0"
        X = X.to(device)
        Y = Y.to(device, non_blocking=True)

        model.eval()
        model.zero_grad()
        X.requires_grad = True

        #turn off gradients for all layers except for the last one to save time
        for name, param in model.named_parameters():
            if name != "output.weight":
                param.require_grad = False
            
            
        logits = model(X)
        losses = F.cross_entropy(logits, Y, reduction='none')
        
        #get loss w.r.t weight gradient for each datapoint
        gradients = []
        for i in range(X.size(0)):

            model.zero_grad()
            losses[i].backward(retain_graph=True)  # retain_graph=True to allow further backward passes
            gradients.append({name : param.grad.clone() for name, param in model.named_parameters()}["output.weight"])
        
        
        #gradients[i][j][k] is gradient of loss w.r.t the jth weight of the kth output neuron for datapoint i
        gradients = torch.stack(gradients)
        gradients = gradients.reshape(gradients.shape[0], gradients.shape[1] * gradients.shape[2])
        
        avg_dset_grad = torch.mean(gradients, dim=0)

        
        GA_scores = []
        for i in range(len(slice_idxs)):
            GA_score = torch.norm(avg_dset_grad - torch.mean(gradients[slice_idxs[i]], dim=0), p=2)
            GA_scores.append(GA_score)
            
        GA_scores = torch.stack(GA_scores)
    
        #turn back on gradients for all layers
        for name, param in model.named_parameters():
                param.require_grad = True
            
            
        return GA_scores
        

class AdultDataset(Dataset):
    def __init__(self, path):
        self.X, self.Y = self.parseData(path)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def parseData(self, f_name):
        with open(f_name, 'rb') as file:
            return pkl.load(file)




class Trainer:
    def __init__(self, trainSet: Dataset, testSet: Dataset, model: SimpleNN, params:dict, seed=1, VOG=True, checkpointFreq = 1):
        self.trainSet = trainSet
        set_seed(seed)
        self.model = model
        self.params = params
        self.train_loader = DataLoader(trainSet, batch_size=5000, num_workers=8, pin_memory=True, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(testSet, batch_size=5000, num_workers=8, pin_memory=True)
        self.metrics = {"train": {"accuracy" : [], "loss" : []}, "test": {"accuracy" : [], "loss" : []}}
        
        self.slice_idxs = get_slice_idxs()

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["lr"], weight_decay=params["weight decay"])
        self.sum_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        
        self.numCheckpoints = (self.params["epochs"] // checkpointFreq) + 1
        self.checkpointFreq = checkpointFreq
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model = self.model.to(self.device)
        
        #finalScores[i] contains the VOG of datapoint i
        self.VOG = RunningVOG((len(trainSet), 128))
        
    def train(self):
          
        for epoch in range(self.params["epochs"]):
            
            self.model.train()
            epoch_grads = None
            
            for i, (features, targets) in enumerate(self.train_loader):
                features = features.to(self.device)
                targets = targets.to(self.device, non_blocking=True)
                features = features.requires_grad_(True)
                
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.loss(logits, targets)
                loss.backward()
                
                pre_softmax_activations = self.model(features)
                pre_softmax_activations.retain_grad()
                
                if epoch % self.checkpointFreq == 0:
                    activation_grad = torch.autograd.grad(
                        outputs=pre_softmax_activations,
                        inputs=features,
                        grad_outputs=torch.ones_like(pre_softmax_activations),  
                        create_graph=False,
                        allow_unused=True
                    )[0]

                    if activation_grad is not None:
                        activation_grad = activation_grad.cpu()
                        if epoch_grads is None:
                            epoch_grads = activation_grad
                        else:
                            epoch_grads = torch.cat((epoch_grads, activation_grad), dim=0)
                
                self.optimizer.step()
            
            if epoch % self.checkpointFreq == 0:
                self.VOG.update(epoch_grads)
                
            self.model.eval()
            self.update_metrics()
            print("epoch " + str(epoch) + " train accuacy=" + str(self.metrics["train"]["accuracy"][-1] ))


            
        self.metrics["train"]["slice vogs"] = self.VOG.get_slice_VOGs(self.slice_idxs)
        self.metrics["train"]["slice GA"] = self.model.get_slice_ga_errors(self.trainSet.X, self.trainSet.Y, self.slice_idxs)
        self.metrics["train"]["slice accs"] = self.model.per_slice_accuracy(self.trainSet.X, self.trainSet.Y, self.slice_idxs)
        self.metrics["train"]["slice loss"] = self.model.per_slice_loss(self.trainSet.X, self.trainSet.Y, self.slice_idxs)

    def get_GA_errors(self, slice_idxs_list):
        per_slice_GA = []
        ts = self.trainSet
        #get each one predicate slice
        for i in range(len(slice_idxs_list)):
            slice_idxs = slice_idxs_list[i]
            slice_ga = self.model.getGAError(ts.X, ts.X[slice_idxs], ts.Y, ts.Y[slice_idxs])
            per_slice_GA.append(slice_ga.item())
           
        return per_slice_GA
    def make_plots(self):
        
        epochs = np.arange(self.params["epochs"])
        QuickPlot([epochs, epochs], [self.metrics["train"]["loss"], self.metrics["test"]["loss"]], ["train loss", "test loss"], "epoch", "avg loss", "Simple deep NN on Adult Loss curve", markLast=True)
        QuickPlot([epochs, epochs], [self.metrics["train"]["accuracy"], self.metrics["test"]["accuracy"]], ["train accuracy", "test accuracy"], "epoch", "accuracy", "Simple deep NN on Adult Accuracy Curve", markLast=True)  
        
    def update_metrics(self):
            for ss in ["train", "test"]:
                correct = 0
                loss = 0
                loader = self.train_loader if ss == "train" else self.test_loader
                
                for X, Y in loader:
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    logits = self.model(X)
                    loss += self.sum_loss(logits, Y).item()  
                    preds = torch.argmax(logits, dim=1)
                    correct += torch.sum(preds == Y).item()


                self.metrics[ss]["accuracy"].append(correct/len(loader.dataset))
                self.metrics[ss]["loss"].append(loss/len(loader.dataset))
       



def test():
    
    seeds = [random.randint(0, 10000000) for _ in range(10)]
    for i in range(10):
        seed = seeds[i]
        
        train_set = AdultDataset("./Data/Adult/train.pkl")
        test_set = AdultDataset("./Data/Adult/test.pkl")
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(train_set.X)
        train_set.X = torch.tensor(encoder.transform(train_set.X), dtype=torch.float32)
        test_set.X = torch.tensor(encoder.transform(test_set.X), dtype=torch.float32)
        

        params = {
            "epochs" : 15,
            "lr" : 0.001,
            "weight decay" : 1e-5
        }
        
        model = SimpleNN(128, 50, 2)
        trainer = Trainer(train_set, test_set, model, params,seed=seed)
        trainer.train()
        
        
        with open("./metrics/" + str(seed) + "metrics.pkl", 'wb') as file:
            pkl.dump(trainer.metrics, file)
        
        with open("./metrics/" + str(seed) + "metrics.pkl", 'rb') as file:
            metrics = pkl.load(file)
            print(metrics)
        
    
    
    
if __name__ == "__main__":
    test()