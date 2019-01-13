# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().magic(u'matplotlib inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |


# download the dataset
get_ipython().system(u'wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


#Load Data From CSV File  
df = pd.read_csv('loan_train.csv')
df.head()

df.shape


#Convert to date time object 
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# Data visualization and pre-processing

df['loan_status'].value_counts()

get_ipython().system(u'conda install -c anaconda seaborn -y')
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# Pre-processing:  Feature selection/extraction

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()



df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# Convert Categorical features to numerical values
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# Lets convert male to 0 and female to 1:
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# One Hot Encoding  
df.groupby(['education'])['loan_status'].value_counts(normalize=True)


#Feature befor One Hot Encoding

df[['Principal','terms','age','Gender','education']].head()


#Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


#Feature selection
X = Feature
X[0:5]


# Finding lables
y = df['loan_status'].values
y[0:5]


#Normalize Data 
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# Classification 
# K Nearest Neighbor(KNN)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


k = 7
neigh7 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat7 = neigh7.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh7.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat7))


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loanTree # it shows the default parameters


loanTree.fit(X_train,y_train)


predTree = loanTree.predict(X_test)


print (predTree [0:5])
print (y_test [0:5])


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


loanTree

get_ipython().system(u'conda install -c conda-forge pydotplus -y')
get_ipython().system(u'conda install -c conda-forge python-graphviz -y')


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().magic(u'matplotlib inline')

dot_data = StringIO()
filename = "loantree.png"
featureNames = Feature.columns[0:8]
targetNames = df["loan_status"].unique().tolist()
out=tree.export_graphviz(loanTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

# Support Vector Machine
from sklearn.metrics import classification_report, confusion_matrix
import itertools


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

yhatSVM = clf.predict(X_test)
yhatSVM [0:5]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhatSVM, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)


print (classification_report(y_test, yhatSVM))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
f1_score(y_test, yhatSVM, average='weighted')


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhatSVM)

 # Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

yhatLR = LR.predict(X_test)
yhatLR


yhatLR_prob = LR.predict_proba(X_test)
yhatLR_prob


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhatLR, labels=['PAIDOFF','COLLECTION']))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhatLR, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')


print (classification_report(y_test, yhatLR))


from sklearn.metrics import log_loss
log_loss(y_test, yhatLR_prob)


# Model Evaluation using Test set

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

get_ipython().system(u'wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# Load Test set for evaluation 
test_df = pd.read_csv('loan_test.csv')
test_df.head()


# Pre-processing: Feature selection/extraction
#Pre-processing for formatting
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.drop(['Unnamed: 0'], axis = 1,inplace=True)
test_df.drop(['Unnamed: 0.1'], axis = 1,inplace=True)

test_df.head()


FeatureTDF = test_df[['Principal','terms','age','Gender','weekend']]
FeatureTDF = pd.concat([FeatureTDF,pd.get_dummies(test_df['education'])], axis=1)
FeatureTDF.rename(columns = {'Bechalor':'bachelor'}, inplace = True)
FeatureTDF.rename(columns = {'Principal':'principal'}, inplace = True)
FeatureTDF.rename(columns = {'Gender':'gender'}, inplace = True)
FeatureTDF.rename(columns = {'High School or Below':'high school or below'}, inplace = True)
FeatureTDF.drop(['Master or Above'], axis = 1,inplace=True)


XTDF = FeatureTDF
XTDF.head()


yTDF = test_df['loan_status'].values
yTDF[0:5]


XTDF = preprocessing.StandardScaler().fit(XTDF).transform(XTDF)
XTDF[0:5]


# ACCURACY

yhatTDF = neigh7.predict(XTDF)
print("TDF set Accuracy: ", metrics.accuracy_score(yTDF, yhatTDF))



predTreeTDF = loanTree.predict(XTDF)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(yTDF, predTreeTDF))



yhatSVMTDF = clf.predict(XTDF)
from sklearn.metrics import f1_score
f1_score(yTDF, yhatSVMTDF, average='weighted')


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(yTDF, yhatSVMTDF)


yhatLRTDF = LR.predict(XTDF)
yhatLRTDF_prob = LR.predict_proba(XTDF)


LR_JCC = jaccard_similarity_score(yTDF, yhatLRTDF)
LR_Log = log_loss(yTDF, yhatLRTDF_prob)
LR_F1 =  f1_score(yTDF,  yhatLRTDF, average='weighted', labels=np.unique(yhatLRTDF)) 
print("Logistic Regression Accuracy", LR_JCC, LR_F1, LR_Log)


#  Report


KNN_JCC = jaccard_similarity_score(yTDF, yhatTDF)
KNN_F1 =  f1_score(yTDF,  yhatTDF, average='weighted', labels=np.unique(yhatTDF)) 

DT_JCC = jaccard_similarity_score(yTDF, predTreeTDF)
DT_F1 =  f1_score(yTDF,  predTreeTDF, average='weighted', labels=np.unique(predTreeTDF)) 


SVM_JCC = jaccard_similarity_score(yTDF, yhatSVMTDF)
SVM_F1 =  f1_score(yTDF,  yhatSVMTDF, average='weighted', labels=np.unique(yhatSVMTDF)) 

LR_JCC = jaccard_similarity_score(yTDF, yhatLRTDF)
LR_Log = log_loss(yTDF, yhatLRTDF_prob)
LR_F1 =  f1_score(yTDF,  yhatLRTDF, average='weighted', labels=np.unique(yhatLRTDF)) 


print("KNN                 ", KNN_JCC, KNN_F1, "NA")
print("Decision Tree       ", DT_JCC, DT_F1, "NA")
print("SVM                 ", SVM_JCC, SVM_F1, "NA")
print("Logistic Regression ", LR_JCC, LR_F1, LR_Log)


Results = {'Algorithm':['KNN','Decision Tree','SVM','Logistic Regression'],
        'Jaccard':[KNN_JCC, DT_JCC,SVM_JCC,LR_JCC],
        'F1-score':[KNN_F1,DT_F1,SVM_F1,LR_F1],
        'LogLoss':['NA','NA','NA',LR_Log]
        }

from pandas import DataFrame
Report = DataFrame(Results, columns= ['Algorithm', 'Jaccard', 'F1-score', 'LogLoss'])

from IPython.display import display, HTML
display(HTML(Report.to_html(index=False)))
