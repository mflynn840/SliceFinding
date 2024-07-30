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
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


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
    




def pickleDataset(X_train, Y_train, X_test, Y_test, path=""):

    with open(os.path.join(path, "train.pkl"), 'wb') as file:
        pkl.dump((X_train, Y_train), file)

    with open(os.path.join(path, "test.pkl"), 'wb') as file:
        pkl.dump((X_test, Y_test), file)
        

        
            

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
    
    print(test_df.columns)
    #MAKE BINS FOR






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
    

    return X_train, Y_train, X_test, Y_test, test_df, train_df


def unpickleDataset(path):
    
    with open(path, 'wb') as file:
        return pkl.load(file)
        
    


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
        

#Load dataset and save its numpy representation to a file
X_train, Y_train, X_test, Y_test, train_df, test_df = prepAdult()
print(train_df.head)





#pickleDataset(X_train, Y_train, X_test, Y_test, path="Data/Adult")



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
