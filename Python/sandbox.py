
import torch

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
    

m=10
n=6
vecs1 = torch.randn(m,n)*100
vecs2 = torch.randn(m,n)*100
#there are 10 datapoints, 6 features
vog = RunningVOG((10, 6))
vog.update(vecs1)
vog.update(vecs2)
mean = (vecs1 + vecs2)/2
print("means")
print(mean)
print()
print(vog.get_mean())


variance_tensor = ((vecs1 - mean) ** 2 + (vecs2 - mean) ** 2) / 1

print("var")
print(variance_tensor)
print()
print(vog.get_varience())


    
