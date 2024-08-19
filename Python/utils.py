import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import logging
import colorlog

class QuickPlot:
    def __init__(self, Xs, Ys, labels, xlab="x", ylab="y", title="Title", markLast=False, percent=False):

        if type(Xs) is list:
            for X, Y, label in zip(Xs, Ys, labels):
                plt.plot(X, Y, label=label, linestyle='-')
                
                if markLast:
                    if percent:
                        plt.text( X[-1],Y[-1], f'Final = {Y[-1]*100:.2f} %', fontsize=14, ha='right', va='bottom')
                    else:
                        plt.text( X[-1],Y[-1], f'Final = {Y[-1]:.3f} ', fontsize=14, ha='right', va='bottom')
        else:
            plt.plot(X, Y)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)


        # Add a legend, grid, and show the plot
        plt.legend(markerscale=1.5)
        plt.grid(True)
        plt.show()
        


def regressionGraph(x1, x2, groupName, ss):
    
    #x1 is accuracy
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2)
    
    model = LinearRegression()
    model.fit(x1,x2)
    

    # Predict values
    x1_pred = np.linspace(min(x1), max(x1), 100).reshape(-1, 1)
    x2_pred = model.predict(x1_pred)

    # Compute Pearson correlation coefficient
    r, _ = pearsonr(x1.flatten(), x2)

    # Get slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_
    # Plot the data and the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, x2, color='blue', label='Data points')
    plt.plot(x1_pred, x2_pred, color='red', linestyle='--', label='Regression line')

    # Annotate Pearson correlation coefficient, slope, and intercept inside the plot area
    plt.annotate(f'Pearson r = {r:.2f}\nSlope = {slope:.2f}\nIntercept = {intercept:.2f}', 
                 xy=(0.1, 0.9), 
                 xycoords='axes fraction', 
                 fontsize=14, 
                 ha='left', 
                 va='top', 
                 color='red',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    
    # Add labels and title
    plt.xlabel('Group Accuracy')
    plt.ylabel('Group GA error')
    plt.title(groupName + " Accuracy vs. GA error (" + ss + " set)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    
          
def visualizeAdult(path = "Data/Adult/adult.data"):
    #get a dataframe of the datasets csv file
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital status", "occupation",
        "relationship", "race", "sex", "cap-gain", "cap-loss", "hours/week", "native country", "target"]


    df = pd.read_csv(path, header=None, na_values = "?", skipinitialspace=True)
    df.columns = features

    #remove any entries containing null values
    df = df.dropna()
    

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))

    # 1. Basic Statistics
    #print("Basic Statistics:")
    #print(df.describe(include='all'))

    # 2. Distribution of Continuous Features
    plt.subplot(2, 2, 1)
    sns.histplot(df['age'], kde=True)
    plt.title('Distribution of Age')

    plt.subplot(2, 2, 2)
    sns.histplot(df['cap-gain'], kde=True)
    plt.title('Distribution of Capital Gain')

    plt.subplot(2, 2, 3)
    sns.histplot(df['cap-loss'], kde=True)
    plt.title('Distribution of Capital Loss')

    plt.subplot(2, 2, 4)
    sns.histplot(df['hours/week'], kde=True)
    plt.title('Distribution of Work Hours / Week')

    plt.tight_layout()
    plt.show()

    # 3. Count Plot for Categorical Features
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='workclass')
    plt.title('Distribution of Workclass')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='occupation')
    plt.title('Distribution of Occupation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    
    #4. count plots for sensitive features
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='race')
    plt.title('Distribution of Race')
    plt.xticks(rotation=45)
    
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='sex')
    plt.title('Distribution of Sex')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    

    # 5. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # 6. Pairplot to visualize relationships between features
    sns.pairplot(df[['age', 'cap-gain', 'cap-loss', 'hours/week', 'target']], hue='target')
    plt.title('Pairplot of Selected Features')
    plt.show()
    




def pickleDataset(X_train, Y_train, X_test, Y_test, path=""):

    with open(os.path.join(path, "train.pkl"), 'wb') as file:
        pkl.dump((X_train, Y_train), file)

    with open(os.path.join(path, "test.pkl"), 'wb') as file:
        pkl.dump((X_test, Y_test), file)
        

        
            

def prepAdult(num_bins=10):
    

    trainpath = "Data/Adult/adult.data"
    testpath = "Data/Adult/adult.test"
    
    #get a dataframe of the datasets csv file
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital status", "occupation",
        "relationship", "race", "sex", "cap-gain", "cap-loss", "hours/week", "native country", "target"]
    f_set = set(features)
    
    train_df = pd.read_csv(trainpath, header=None, na_values = "?", skipinitialspace=True)
    test_df = pd.read_csv(testpath, header=None, na_values = "?", skipinitialspace=True)
    train_df.columns = features
    test_df.columns = features
    
    
    #drop duplicate/ very skewed features
    train_df.drop(['education', 'cap-gain', 'cap-loss'], axis=1, inplace=True)
    test_df.drop(['education', 'cap-gain', 'cap-loss'], axis=1, inplace=True)
    
    #remove any entries containing null values and duplicates
    train_df.replace('?', np.nan, inplace=True)
    test_df.replace('?', np.nan, inplace=True)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)
    
    # Remove unwanted character (.) from labels in test set
    test_df["target"] = test_df["target"].str.replace('.', '', regex=False)
    

    #one hot encode categorical features
    categorical_features = ["workclass", "marital status", "occupation", "relationship", "race", "sex", "native country"]
    combined_df = pd.concat([train_df, test_df], keys=["train", "test"])
    combined_df = pd.get_dummies(combined_df, columns = categorical_features)
    
    #bin continuous features
    continuous_features = ["age", "fnlwgt", "hours/week"]
    for feature in continuous_features:
        combined_df[feature] = pd.cut(combined_df[feature], bins=num_bins, labels=np.arange(1,num_bins+1))

        
    #encode targets
    le = LabelEncoder()
    combined_df["target"] = le.fit_transform(combined_df["target"])
    


    #split back into test and train
    train_df = combined_df.xs("train")
    test_df = combined_df.xs("test")
    

    # Reset index after one-hot encoding and splitting
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    #split off the targets and features
    X_train = train_df.drop('target', axis=1).values
    Y_train = train_df['target'].values
    
    X_test = test_df.drop('target', axis=1).values
    Y_test = test_df['target'].values


    return X_train, Y_train, X_test, Y_test, test_df, train_df


def unpickleDataset(train_path, test_path):
    
    X_train, Y_train, X_test, Y_test = None, None, None, None
    
    with open(train_path, 'rb') as file:
        X_train, Y_train = pkl.load(file)
    
    with open(test_path, 'rb') as file:
        X_test, Y_test = pkl.load(file)
    
    return X_train, Y_train, X_test, Y_test
        
        
    
#X_train, Y_train, X_test, Y_test, _, _ = prepAdult()
#pickleDataset(X_train, Y_train, X_test, Y_test, "./Data/Adult")

#X1, Y1, X2, Y2 =  unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")


def getMetrics(X_train, Y_train, X_test, Y_test, model, metrics=None, last=False):
    
    #get a new metrics if one is not provided
    if metrics == None:
        metrics = {
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
                        
    train_preds = model.thresholdedPredict(X_train)
    test_preds = model.thresholdedPredict(X_test)
    
    idxs = None
    with open("Data/Adult/groups.idx", 'rb') as file:
        idxs = pkl.load(file)
    
    
    acc_map = {
        "black male" : "bm_acc",
        "black female" : "bf_acc",
        "white male" : "wm_acc",
        "white female" : "wf_acc",
    }
    
    ga_map = {
        "black male" : "bm_ga",
        "black female" : "bf_ga",
        "white male" : "wm_ga",
        "white female" : "wf_ga"
    }
    
    
    #get all metrics for all groups in train and test sets
    for ss in ["test", "train"]:
        if ss == "test":
            cur_dat = X_test
            cur_preds = test_preds
            cur_truths = Y_test
        else:
            cur_dat = X_train
            cur_preds = train_preds
            cur_truths = Y_train
            
        for group in ["black male", "black female", "white male", "white female"]:
            group_acc = accuracy_score(cur_truths[list(idxs[ss][group])], cur_preds[list(idxs[ss][group])])
            
            #GA error between full dataset and current group subset
            group_ga = model.GAError(cur_dat, cur_truths, cur_dat[list(idxs[ss][group])], Y_train[list(idxs[ss][group])])
            metrics[ss][acc_map[group]].append(group_acc)
            metrics[ss][ga_map[group]].append(group_ga)
            

        #get average accuracy for whole dataset
        metrics[ss]["avg_acc"].append(accuracy_score(cur_truths, cur_preds))


    return metrics
        



def getLogger(progName, fname):
    
    # Configure the logging system
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format
    )
    

    # Create a file handler for plain text output
    fhandler = logging.FileHandler(fname)

    # Define a formatter for the file
    file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    fhandler.setFormatter(file_formatter)
    
    
    #add handlers to logger
    logger = logging.getLogger(progName)

    logger.addHandler(fhandler)
    
    return logger

def is_one_hot(column):
    unique_vals = np.unique(column)
    return set(unique_vals).issubset({0,1})

    

