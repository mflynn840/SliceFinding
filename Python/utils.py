import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class QuickPlot:
    def __init__(self, Xs, Ys, labels, xlab="x", ylab="y", title="Title"):

        if type(Xs) is list:
            for X, Y, label in zip(Xs, Ys, labels):
                plt.plot(X, Y, label=label, linestyle='-')
        else:
            plt.plot(X, Y)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)


        # Add a legend, grid, and show the plot
        plt.legend()
        plt.grid(True)
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
    print("Basic Statistics:")
    print(df.describe(include='all'))

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
    


        
'''
1. Take the CSV file for the train set and
    a. form bins for continous features
    b. make encoders for categorical features

2. On the train and test set
    a. apply the bins and encoders
    b. remove null values

3. seperate the labels from the data

Return X_train, Y_train, X_test, Y_test

'''

def prepAdult():
    

    trainpath = "Data/Adult/adult.data"
    testpath = "Data/Adult/adult.test"
    
    #get a dataframe of the datasets csv file
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital status", "occupation",
        "relationship", "race", "sex", "cap-gain", "cap-loss", "hours/week", "native country", "target"]
    
    train_df = pd.read_csv(trainpath, header=None, na_values = "?", skipinitialspace=True)
    test_df = pd.read_csv(testpath, header=None, na_values = "?", skipinitialspace=True)
    train_df.columns = features
    test_df.columns = features
    
    
    train_df.replace('?', np.nan, inplace=True)
    test_df.replace('?', np.nan, inplace=True)
    #remove any entries containing null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    train_df.drop(['education', 'cap-gain', 'cap-loss'], axis=1, inplace=True)
    test_df.drop(['education', 'cap-gain', 'cap-loss'], axis=1, inplace=True)
    
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)


    
    # Remove unwanted character (.) from labels in test set
    test_df["target"] = test_df["target"].str.replace('.', '', regex=False)
    
    '''
    #find bins for continuous features using the train set
    num_bins=15
    bins_dict = {}
    cont_features = ["age", "fnlwgt", "cap-gain", "cap-loss", "hours/week"]
    for feature in cont_features:
        _, bin_edges = pd.cut(train_df[feature], bins=num_bins, labels=False, retbins=True)
        bins_dict[feature] = bin_edges
    
    
    #apply the bins to the train and test set
    for feature in cont_features:
        train_df[feature] = pd.cut(train_df[feature], bins=bins_dict[feature], labels=False)
        test_df[feature] = pd.cut(test_df[feature], bins=bins_dict[feature], labels=False)
        
    '''

    #one hot encode categorical features
    categorical_features = ["workclass", "marital status", "occupation", "relationship", "race", "sex", "native country"]
    combined_df = pd.concat([train_df, test_df], keys=["train", "test"])
    combined_df = pd.get_dummies(combined_df, columns = categorical_features)
    combined_df["target"] = LabelEncoder().fit_transform(combined_df["target"])
    

    #split back into test and train, convert boolean one hots to ints
    train_df = combined_df.xs("train")
    test_df = combined_df.xs("test")

    
    
    #split off the targets and features
    X_train = train_df.drop('target', axis=1).values
    Y_train = train_df['target'].values
    
    X_test = test_df.drop('target', axis=1).values
    Y_test = test_df['target'].values


    #normalize only the numeric features

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    return X_train, Y_train, X_test, Y_test


def pickleDataset(X_train, Y_train, X_test, Y_test):


    with open("Data/Adult/train.pkl", 'wb') as file:
        pkl.dump((X_train, Y_train), file)

    with open("Data/Adult/test.pkl", 'wb') as file:
        pkl.dump((X_test, Y_test), file)
        

            

def accuracy(preds, ys):
    assert(len(preds) == len(ys))
    return np.sum(preds == ys) / len(preds) * 100
            

sensitiveFeatureIndexes = {
    "isMale" : 44,
    "isFemale" : 43,
    "isBlack" : 40,
    "isWhite" : 42
}



        
#do data preprocessing
X_train, Y_train, X_test, Y_test = prepAdult()

#pickleDataset(X_train, Y_train, X_test, Y_test)






'''
#get a set of encoders for categorical features using train set
encoders = {}
for feature in train_df.columns:
    if train_df[feature].dtype == 'object':
        le = LabelEncoder()
        train_df[feature] = le.fit_transform(train_df[feature])
        encoders[feature] = le


#encode categorical features on test using the same encoders
for feature, encoder in encoders.items():
    test_df[feature] = encoder.transform(test_df[feature])
'''    


'''
    #get indexes of sensitve groups
    index_dict = {
        'test': {
            'male_idx': test_df.index[test_df["sex_Male"] == True].tolist(),
            'female_idx': test_df.index[test_df["sex_Female"] == True].tolist(),
            'white_idx': test_df.index[test_df["race_White"] == True].tolist(),
            'black_idx': test_df.index[test_df["race_Black"] == True].tolist()
        },
        'train': {
            'male_idx': train_df.index[train_df["sex_Male"] == True].tolist(),
            'female_idx': train_df.index[train_df["sex_Female"] == True].tolist(),
            'white_idx': train_df.index[train_df["race_White"] == True].tolist(),
            'black_idx': train_df.index[train_df["race_Black"] == True].tolist()
        }
    
    }
    
    with open("Data/Adult/senstiveAttr.idx", 'wb') as file:
        pkl.dump(index_dict, file)
    
    '''
