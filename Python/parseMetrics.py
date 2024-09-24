import os
import pickle as pkl
from PyTorch import AdultDataset
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import regressionGraph



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def regressionGrid(x1, x2, ss, c_list=None, xlab="", ylab="", show=True):
    # x1 and x2 are matrices with shape (n_samples, n_features)
    num_features = x1.shape[1]
    
    # Determine the grid size
    num_cols = 3  # Number of columns in the grid (adjust as needed)
    num_rows = int(np.ceil(num_features / num_cols))  # Number of rows based on number of features
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 5 * num_rows))
    
    for i in range(num_features):
        x1_col = x1[:, i].reshape(-1, 1)
        x2_col = x2[:, i]

        # Create subplot
        plt.subplot(num_rows, num_cols, i + 1)
        
        model = LinearRegression()
        model.fit(x1_col, x2_col)

        # Predict values
        x1_pred = np.linspace(min(x1_col), max(x1_col), 100).reshape(-1, 1)
        x2_pred = model.predict(x1_pred)

        # Compute Pearson correlation coefficient
        r, _ = pearsonr(x1_col.flatten(), x2_col)

        # Get slope and intercept
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Plot the data
        if c_list is None:
            cmap = plt.get_cmap('viridis')
            scatter = plt.scatter(x1_col, x2_col, c=x2_col, label='Slice', cmap=cmap, s=50)
            
            c_list = []
            colors = scatter.to_rgba(x2_col)
            for color in colors:
                c_list.append(color)
        else:
            plt.scatter(x1_col, x2_col, c=c_list, label='Slice')
        
        plt.plot(x1_pred, x2_pred, color='red', linestyle='--', label='Regression line')

        # Annotate the plot
        plt.annotate(f'Pearson r = {r:.2f}\nSlope = {slope:.2f}\nIntercept = {intercept:.2f}', 
                     xy=(0.1, 0.9), 
                     xycoords='axes fraction', 
                     fontsize=10, 
                     ha='left', 
                     va='top', 
                     color='red',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Add labels and title for each subplot
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f'Run {i + 1}')
        plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    if show:
        plt.show()
    return c_list

def parseMetrics():
    model_runs = {}
    for fname in os.listdir("./metrics"):
        with open(os.path.join("./metrics", fname), 'rb') as file:
            model_runs[fname] = pkl.load(file)
            
    
    train_set = AdultDataset("./Data/Adult/train.pkl")
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(train_set.X)
    train_set.X = torch.tensor(encoder.transform(train_set.X), dtype=torch.float32)


    slice_vogs_per_run = []
    slice_GA_error_per_run = []
    
    for run in model_runs.keys():
        
        slice_vogs = []
        #for each 1-predicate slice
        for i in train_set.X.T:
            slice_idxs = np.where(i == 1)[0]
            #print(model_runs[run]["train"]["point vogs"][slice_idxs].numpy())
            slice_vog = np.mean(model_runs[run]["train"]["point vogs"][slice_idxs].numpy())
            slice_vogs.append(slice_vog)
        slice_vogs_per_run.append(np.asarray(slice_vogs))
        slice_GA_error_per_run.append(model_runs[run]["train"]["point GA"])
        
    all_vogs = np.stack(slice_vogs_per_run)
    all_GAs = np.stack(slice_GA_error_per_run)
    
    print(all_vogs.shape)
    mean_vogs = np.mean(all_vogs, axis=0)
    varience_vogs = np.var(all_vogs, axis=0)
    
    mean_GA = np.mean(all_GAs, axis=0)
    varience_GA = np.var(all_GAs, axis=0)
    
    print("vog mean")
    print(mean_vogs)
    print("vog var")
    print(np.mean(varience_vogs))
    
    print("GA var")
    print(np.mean(varience_GA))
    
    
    


    regressionGrid(all_vogs.T, all_GAs.T, "Model run " + str(i), xlab="Slice VOG", ylab="Slice GA Error")
        
    

    data_for_vog_boxplot = [all_vogs[:,i] for i in range(all_vogs.shape[1])]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_for_vog_boxplot)
    plt.xlabel('Slice Index')
    plt.ylabel('VOG Score')
    plt.title('Boxplot of VOG Scores Across Model Runs')
    # Adjust x-axis ticks
    num_slices = len(data_for_vog_boxplot)
    plt.xticks(ticks=np.arange(0, num_slices, 10), labels=np.arange(0, num_slices, 10))
    plt.show()
    
    data_for_ga_boxplot = [all_GAs[:,i] for i in range(all_GAs.shape[1])]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_for_ga_boxplot)
    plt.xlabel('Slice Index')
    plt.ylabel('GA Score')
    plt.title('Boxplot of GA Scores Across Model Runs')
    # Adjust x-axis ticks
    num_slices = len(data_for_ga_boxplot)
    plt.xticks(ticks=np.arange(0, num_slices, 10), labels=np.arange(0, num_slices, 10))
    plt.show()
    
#parseMetrics()
    
    
    
def testing():
    metrics = None
    slice_idx_list = None
    with open(os.path.join("./metrics", "1730927metrics.pkl"), 'rb') as file:
            metrics = pkl.load(file)
    
    with open("./Python/slice_idxs_list.pkl", 'rb') as file:
        slice_idx_list = pkl.load(file)
        

    print(metrics["train"].keys())


    regressionGraph(metrics["train"]["slice vogs"], metrics["train"]["slice loss"], "train", xlab="slice VOG", ylab="slice loss")
    regressionGraph(metrics["train"]["slice GA"].cpu(), metrics["train"]["slice loss"], "train", xlab="slice GA-error", ylab="slice loss")
    regressionGraph(metrics["train"]["slice accs"], metrics["train"]["slice loss"], "train", xlab="slice accuracy", ylab="slice loss")
    regressionGraph(metrics["train"]["slice GA"].cpu(), metrics["train"]["slice vogs"], "train", xlab="GA error", ylab="VOG")
    
testing()