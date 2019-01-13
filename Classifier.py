
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


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

# Lets download the dataset

# In[2]:


get_ipython().system(u'wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system(u'conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


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


# In[23]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[24]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[25]:


k = 7
neigh7 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat7 = neigh7.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh7.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat7))


# # Decision Tree

# In[26]:


from sklearn.tree import DecisionTreeClassifier

loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loanTree # it shows the default parameters


# In[27]:


loanTree.fit(X_train,y_train)


# In[28]:


predTree = loanTree.predict(X_test)


# In[29]:


print (predTree [0:5])
print (y_test [0:5])


# In[30]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[31]:


loanTree


# In[32]:


get_ipython().system(u'conda install -c conda-forge pydotplus -y')
get_ipython().system(u'conda install -c conda-forge python-graphviz -y')


# In[40]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().magic(u'matplotlib inline')


# In[39]:


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


# # Support Vector Machine

# In[41]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[42]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[43]:


yhatSVM = clf.predict(X_test)
yhatSVM [0:5]


# In[44]:


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


# In[45]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhatSVM, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)


print (classification_report(y_test, yhatSVM))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')


# In[46]:


from sklearn.metrics import f1_score
f1_score(y_test, yhatSVM, average='weighted')


# In[47]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhatSVM)


# # Logistic Regression

# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[49]:


yhatLR = LR.predict(X_test)
yhatLR


# In[50]:


yhatLR_prob = LR.predict_proba(X_test)
yhatLR_prob


# In[51]:


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


# In[52]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhatLR, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')


# In[53]:


print (classification_report(y_test, yhatLR))


# In[54]:


from sklearn.metrics import log_loss
log_loss(y_test, yhatLR_prob)


# # Model Evaluation using Test set

# In[34]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[35]:


get_ipython().system(u'wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[36]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# Pre-processing: Feature selection/extraction

# In[55]:


##Pre-processing for formatting
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.drop(['Unnamed: 0'], axis = 1,inplace=True)
test_df.drop(['Unnamed: 0.1'], axis = 1,inplace=True)


# In[56]:


test_df.head()


# In[57]:


FeatureTDF = test_df[['Principal','terms','age','Gender','weekend']]
FeatureTDF = pd.concat([FeatureTDF,pd.get_dummies(test_df['education'])], axis=1)
FeatureTDF.rename(columns = {'Bechalor':'bachelor'}, inplace = True)
FeatureTDF.rename(columns = {'Principal':'principal'}, inplace = True)
FeatureTDF.rename(columns = {'Gender':'gender'}, inplace = True)
FeatureTDF.rename(columns = {'High School or Below':'high school or below'}, inplace = True)
FeatureTDF.drop(['Master or Above'], axis = 1,inplace=True)


# In[58]:


XTDF = FeatureTDF
XTDF.head()


# In[59]:


yTDF = test_df['loan_status'].values
yTDF[0:5]


# In[60]:


XTDF = preprocessing.StandardScaler().fit(XTDF).transform(XTDF)
XTDF[0:5]


# ACCURACY

# In[61]:


yhatTDF = neigh7.predict(XTDF)
print("TDF set Accuracy: ", metrics.accuracy_score(yTDF, yhatTDF))


# In[62]:


predTreeTDF = loanTree.predict(XTDF)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(yTDF, predTreeTDF))


# In[63]:


yhatSVMTDF = clf.predict(XTDF)
from sklearn.metrics import f1_score
f1_score(yTDF, yhatSVMTDF, average='weighted')


# In[64]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(yTDF, yhatSVMTDF)


# In[65]:


yhatLRTDF = LR.predict(XTDF)
yhatLRTDF_prob = LR.predict_proba(XTDF)


# In[66]:


LR_JCC = jaccard_similarity_score(yTDF, yhatLRTDF)
LR_Log = log_loss(yTDF, yhatLRTDF_prob)
LR_F1 =  f1_score(yTDF,  yhatLRTDF, average='weighted', labels=np.unique(yhatLRTDF)) 
print("Logistic Regression Accuracy", LR_JCC, LR_F1, LR_Log)


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# In[67]:


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


# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
