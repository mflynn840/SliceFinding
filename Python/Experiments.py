from SliceLine import SliceLine
from utils import unpickleDataset
from Model import LogisticModel
import math
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from SliceLine import SliceLine
import pickle as pkl
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


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

    print(domains)
    #train a logistic regression model and get the per example error
    model = LogisticModel(X_train_one_hot.shape[1])
    model.fit(X_train_one_hot, Y_train, X_test_one_hot, Y_test, 20, .01)
    e = model.per_example_error(X_train_one_hot, Y_train)
    sigma = max(math.ceil(0.01 * X_train.shape[0]), 8)


    #run sliceline
    sf = SliceLine(X_train, e, k=10, sigma=sigma, alpha=0.95, L=4, auto=True)
    
    print(sf.result)
    SliceLine.pretty_print_results(sf.result, featureNames, domains)
    
    
def VOGvGA():
    X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")

    encoder = OneHotEncoder()
    encoder.fit(X_train)
    X_train = encoder.transform(X_train).toarray()
    X_test = encoder.transform(X_test).toarray()
    model = LogisticModel(128, VOG=True)
    model.fit(X_train, Y_train, X_test, Y_test, 200, .01, showMetrics=False)
    
    VOG = model.VOG
    VOGs = []
    for i in X_train.T:
        slice_idxs = np.where(i == 1)[0]
        VOGs.append(np.mean(VOG[slice_idxs]))


    metrics = model.metrics
    accs_early = [] 
    accs_late = []
    for i in metrics["train"].keys():
        if "acc" in i and i != "avg_acc":
            accs_early.append(metrics["train"][i][49])
            accs_late.append(metrics["train"][i][199])
            
     
    gas_early = []   
    gas_late = []    
    for i in metrics["train"].keys():
        if "ga" in i:
            gas_early.append(metrics["train"][i][49])
            gas_late.append(metrics["train"][i][199])

            
    colors1 = regressionGraph(VOGs, accs_early, "50", xlab="Slice VOG", ylab="slice accuracy")
    colors2 = regressionGraph(VOGs, gas_early, "50", xlab="Slice VOG", ylab = "Slice GA error")
    
    regressionGraph(VOGs, accs_late, "200", c_list=colors1, xlab="Slice VOG", ylab="slice accuracy")
    regressionGraph(VOGs, gas_late, "200", c_list=colors2, xlab="Slice VOG", ylab = "Slice GA error")
    
        
def score(alpha, avg_dset_err, slice_sizes, slice_errors, total_samples):
    slice_sizes = np.asarray(slice_sizes)
    slice_errors = np.asarray(slice_errors)
    sc = alpha * ((slice_errors/slice_sizes) / avg_dset_err - 1) - (1-alpha) * (total_samples/slice_sizes - 1)
    return np.nan_to_num(sc, nan=-np.inf)


def VOGvScores(earlyEpoch=49, lateEpoch=199, numEpochs=200):
    X_train, Y_train, X_test, Y_test = unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")

    encoder = OneHotEncoder()
    encoder.fit(X_train)
    X_train_new = encoder.transform(X_train).toarray()
    X_test = encoder.transform(X_test).toarray()
    
    
    dataset_size = X_train.shape[0]
    model = LogisticModel(128, VOG=True)
    model.fit(X_train_new, Y_train, X_test, Y_test, numEpochs, .01, showMetrics=False)

    VOG = model.VOG
    VOGs = []
    
    sizes = []
    for i in X_train_new.T:
        slice_idxs = np.where(i == 1)[0]
        VOGs.append(np.mean(VOG[slice_idxs]))
        sizes.append(len(slice_idxs))
    

    metrics = model.metrics
    accs_early = [] 
    accs_late = []
    for i in metrics["train"].keys():
        if "acc" in i and i != "avg_acc":
            accs_early.append(metrics["train"][i][earlyEpoch])
            accs_late.append(metrics["train"][i][lateEpoch])


    scores_early = {}
    scores_late = {}
    
    for alpha in [0, .15, .25, .35, .45, .55, .65, .75, .85, .9, .95, 1.0]:
        scores_early[alpha] = score(alpha, metrics["train"]["avg_acc"][earlyEpoch], sizes, accs_early, dataset_size)
        scores_late[alpha] = score(alpha, metrics["train"]["avg_acc"][lateEpoch], sizes, accs_late, dataset_size)
    

    
    gas_early = []   
    gas_late = []    
    for i in metrics["train"].keys():
        if "ga" in i:
            gas_early.append(metrics["train"][i][earlyEpoch])
            gas_late.append(metrics["train"][i][lateEpoch])

    
    rs_early = {}
    
    for alpha in scores_early.keys():
        r, _ = pearsonr(VOGs, scores_early[alpha])
        regressionGraph(VOGs, scores_early[alpha], "50", xlab="Slice VOG", ylab="SliceLine Score (alpha =" + str(alpha) + ")")
        rs_early[alpha] = r
        

    #find the alpha with best r value
    best_alpha = max(rs_early, key=lambda k: abs(rs_early[k]))
    worst_alpha = max(rs_early, key=lambda k: abs(rs_early[k]))
    
    colors_sl, r_sl = regressionGraph(VOGs, scores_early[best_alpha], "50", xlab="Slice VOG", ylab="SliceLine Score (alpha =" + str(best_alpha) + ")")
    colors_sl, r_sl = regressionGraph(VOGs, scores_early[worst_alpha], "50", xlab="Slice VOG", ylab="SliceLine Score (alpha =" + str(worst_alpha) + ")")
    colors_ga, r_ga = regressionGraph(VOGs, gas_early, "50", xlab="Slice VOG", ylab = "Slice GA error")
    
    regressionGraph(VOGs, scores_late[best_alpha], "200", c_list=colors_sl, xlab="Slice VOG", ylab="SliceLine Score (alpha =" + str(best_alpha) + ")" )
    regressionGraph(VOGs, gas_late, "200", c_list=colors_ga, xlab="Slice VOG", ylab = "Slice GA error")

    
    print(best_alpha)
