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
                plt.text( X[-1],Y[-1], f'Final = {Y[-1]*100:.2f} %', fontsize=14, ha='right', va='bottom')
        else:
            plt.plot(X, Y)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)


        # Add a legend, grid, and show the plot
        plt.legend(markerscale=1.5)
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

    #normalize the numeric features
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
            

def getMetrics(X_train, Y_train, X_test, Y_test, model, metrics=None):
    
    if metrics == None:
        metrics = {
                    "train": {
                        "bm_acc": [],
                        "bf_acc": [],
                        "wm_acc": [],
                        "wf_acc": [],
                        "avg_acc" : [],
                    },
                    "test": {
                        "bm_acc": [],
                        "bf_acc": [],
                        "wm_acc": [],
                        "wf_acc": [],
                        "avg_acc" : [],
                    }
                }
                        
    train_preds = model.thresholdedPredict(X_train)
    test_preds = model.thresholdedPredict(X_test)
    
    idxs = None
    with open("Data/Adult/groups.idx", 'rb') as file:
        idxs = pkl.load(file)

    

    train_bm_acc = accuracy_score(Y_train[list(idxs["test"]["black male"])], train_preds[list(idxs["test"]["black male"])])
    train_bf_acc = accuracy_score(Y_train[list(idxs["test"]["black female"])], train_preds[list(idxs["test"]["black female"])])
    train_wm_acc = accuracy_score(Y_train[list(idxs["test"]["white male"])], train_preds[list(idxs["test"]["white male"])])
    train_wf_acc = accuracy_score(Y_train[list(idxs["test"]["white female"])], train_preds[list(idxs["test"]["white female"])])

    test_bm_acc = accuracy_score(Y_test[list(idxs["test"]["black male"])], test_preds[list(idxs["test"]["black male"])])
    test_bf_acc = accuracy_score(Y_test[list(idxs["test"]["black female"])], test_preds[list(idxs["test"]["black female"])])
    test_wm_acc = accuracy_score(Y_test[list(idxs["test"]["white male"])], test_preds[list(idxs["test"]["white male"])])
    test_wf_acc = accuracy_score(Y_test[list(idxs["test"]["white female"])], test_preds[list(idxs["test"]["white female"])])


    metrics["train"]["bm_acc"].append(train_bm_acc)
    metrics["train"]["bf_acc"].append(train_bf_acc)
    metrics["train"]["wm_acc"].append(train_wm_acc)
    metrics["train"]["wf_acc"].append(train_wf_acc)
    metrics["train"]["avg_acc"].append(accuracy_score(Y_train, train_preds))

    metrics["test"]["bm_acc"].append(test_bm_acc)
    metrics["test"]["bf_acc"].append(test_bf_acc)
    metrics["test"]["wm_acc"].append(test_wm_acc)
    metrics["test"]["wf_acc"].append(test_wf_acc)
    metrics["test"]["avg_acc"].append(accuracy_score(Y_test, test_preds))

    return metrics
    


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
    #get indicies of sensitve groups
    index_dict = {
        'test': {
            'male_idx': np.where(test_df["sex_Male"].values)[0],
            'female_idx': np.where(test_df["sex_Female"].values)[0],
            'white_idx': np.where(test_df["race_White"].values)[0],
            'black_idx': np.where(test_df["race_Black"].values)[0]
        },
        'train': {
            'male_idx': np.where(train_df["sex_Male"].values)[0],
            'female_idx': np.where(train_df["sex_Female"].values)[0],
            'white_idx': np.where(train_df["race_White"].values)[0],
            'black_idx': np.where(train_df["race_Black"].values)[0],
        }
    
    }
    
    group_dict = {
        'train': {
            'black male': set(index_dict['train']['black_idx']).intersection(index_dict['train']['male_idx']),
            'black female': set(index_dict['train']['black_idx']).intersection(index_dict['train']['female_idx']),
            'white male': set(index_dict['train']['white_idx']).intersection(index_dict['train']['male_idx']),
            'white female': set(index_dict['train']['white_idx']).intersection(index_dict['train']['female_idx'])
        },
        'test': {
            'black male': set(index_dict['test']['black_idx']).intersection(index_dict['test']['male_idx']),
            'black female': set(index_dict['test']['black_idx']).intersection(index_dict['test']['female_idx']),
            'white male': set(index_dict['test']['white_idx']).intersection(index_dict['test']['male_idx']),
            'white female': set(index_dict['test']['white_idx']).intersection(index_dict['test']['female_idx'])
        }
    }
    
    with open("Data/Adult/groups.idx", 'wb') as file:
        pkl.dump(group_dict, file)
        
        

    
'''




'''
    
    print("Number of Black Males (Train Set): " + str(len(list(idxs["train"]["black male"]))))
    print("Number of Black Females (Train Set): " + str(len(list(idxs["train"]["black female"]))))
    print("Number of White Males (Train Set): " + str(len(list(idxs["train"]["white male"]))))
    print("Number of White Females (Train Set): " + str(len(list(idxs["train"]["white female"]))))

    # Print the number of instances for each group in the test set
    print("Number of Black Males (Test Set): " + str(len(list(idxs["test"]["black male"]))))
    print("Number of Black Females (Test Set): " + str(len(list(idxs["test"]["black female"]))))
    print("Number of White Males (Test Set): " + str(len(list(idxs["test"]["white male"]))))
    print("Number of White Females (Test Set): " + str(len(list(idxs["test"]["white female"]))))

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
        
        idxs = None
    with open("Data/Adult/senstiveAttr.idx", 'rb') as file:
        idxs = pkl.load(file)
    
    assert( idxs != None)
    train_male = set(idxs["train"]["male_idx"])
    train_female = set(idxs["train"]["female_idx"])
    train_black = set(idxs["train"]["black_idx"])
    train_white = set(idxs["train"]["white_idx"])
    
    test_male = set(idxs["test"]["male_idx"])
    test_female = set(idxs["test"]["female_idx"])
    test_black = set(idxs["test"]["black_idx"])
    test_white = set(idxs["test"]["white_idx"])
    
    
    newDict = {
        "train" : {
            "black male" : train_male.intersection(train_black),
            "black female" : train_female.intersection(train_black),
            "white male" : train_male.intersection(train_white),
            "white female" : train_female.intersection(train_white)
        },
        
        "test" : {
            "black male": test_male.intersection(test_black),
            "black female": test_female.intersection(test_black),
            "white male": test_male.intersection(test_white),
            "white female": test_female.intersection(test_white)
            
        }
    }
    
    with open("Data/Adult/GenderSex.idx", 'wb') as file:
        pkl.dump(newDict, file)
    
    
    '''
    
    

'''
    demographic_groups = {
        'white male': (test_df['race_White'] == 1) & (test_df['sex_Male'] == 1),
        'white female': (test_df['race_White'] == 1) & (test_df['sex_Female'] == 1),
        'black male': (test_df['race_Black'] == 1) & (test_df['sex_Male'] == 1),
        'black female': (test_df['race_Black'] == 1) & (test_df['sex_Female'] == 1),
        # Add other demographic groups as needed
    }
    

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration
    
    for ax, (group, condition) in zip(axes, demographic_groups.items()):
        group_df = test_df[condition]
        total_count = len(group_df)
        
        if total_count == 0:
            continue  # Skip groups with no members
        
        # Count the number of individuals in each class
        class_0_count = np.sum(group_df["target"] == 0)
        class_1_count = np.sum(group_df["target"] == 1)
        
        # Calculate percentages
        sizes = [class_0_count / total_count, class_1_count / total_count]
        labels = ['<=50k', '>50k']
        
        # Plot pie chart
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'Distribution of {group}')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    


'''
