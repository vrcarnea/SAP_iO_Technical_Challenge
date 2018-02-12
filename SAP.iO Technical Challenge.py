# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:11:26 2018

@author: Victoria
"""

import pandas as pd
import os 
import matplotlib.pyplot as plt

#Set a working directory for ease of code assimilation for user
os.chdir('C:\\Users\Victoria\Documents\MSA\SAP.io')

#Read in dataset
df = pd.read_csv('SAPio_DataScience_Challenge[1].csv')

df.type =  df.type.astype('category')

################# Exploratory Data Analysis #################

#First check columns for missing values and see how many are missing
df[df.columns[df.isnull().any()]].isnull().sum()

#5 columns have missing values. Volatile Acidity, Astringency Rating, Residual Sugar, pH, and Vintage
#The most concerning is the residual sugar column it is missing 2364 values, about 36% of the data is missing.
#Need to consider imputation methods

#Check the distribution of the 'response' -- quality

df.quality.value_counts()
df.quality.plot.hist(bins = 7)

#Both of these code lines produce results that show how the quality variable is distributed.
#There are no values below 3 or above 9, most falling at 6. Can use this information to inform decision making

#Because of the large class imbalance and the natire of the project -- What makes wine 'good'? -- I made the assumption that anything with a quality greater than 6 good
df.loc[(df.quality == 3) | (df.quality == 4) | (df.quality == 5) | (df.quality == 6),'quality_cat'] = 'Not Good'
df.loc[(df.quality == 7) | (df.quality == 8) | (df.quality == 9) ,'quality_cat'] = 'Good'

df.quality_cat =  df.quality_cat.astype('category')

df.quality_cat.value_counts()
#The response is still class imbalanced -- less than 25% of the data is classified as good. May need to oversample to get good results.

# Now look at the rest of the variables

#Quickly look at the distribution of the continuous variables
for i in range(1,13):
    plt.figure();
    df.iloc[:,i].plot.hist()
    plt.title(df.columns[i])

for i in range(1,13):
    print(df.columns[i])
    for j in range (1,13):
        print(df.iloc[:,i].corr(df.iloc[:,j]),df.columns[j])

pd.options.display.mpl_style = 'default'
df.boxplot(by='quality_cat')

#Tese boxplots are helpful but unseeable if I had more time I would imporve their print out

#Appears that pretty much all of the distributions are left skewed, again this will only present a problem depending on how I proceed
#Not all of them are severely skewed so any transformations may not be necessary for all of the varaibles

#I will be treating the vintage variable as a factor, due to the nature of its meaning in this context
df.vintage.value_counts()

#There is only one observtion with the 2001 vintage, probably can't make any assertions for the 2001 vintage based on this dataset
#This will most likely cause a problem when I run a model, therefore I will create dummies to elminate this issue as well as the missing values

df.type.value_counts()

#There are more observations for white wine than for red wine 

#My initial reaction to this problem is to develop a Decision Tree model to help begin explain what makes up a 'good' wine

################# Data Manipulation for Models #################
#Create a Binary Response -- 1 : Good and 0 : Not Good
df.loc[(df.quality == 3) | (df.quality == 4) | (df.quality == 5) | (df.quality == 6),'quality_bin'] = 0
df.loc[(df.quality == 7) | (df.quality == 8) | (df.quality == 9) ,'quality_bin'] = 1

#Set up the data 
X = df.iloc[:,0:13]
Y = df.quality_bin

#Create the bound sulfur dioxide variable as well as the ratio and check their dists
X['bounded_so2'] = X['total sulfur dioxide'] - X['free sulfur dioxide'] 
X['so2_ratio'] = X['free sulfur dioxide'] / X['bounded_so2']

#Created a Bounded SO2 variable as well as a ratio variable in hopes to gain more information from the total SO2.
#Did some quick research and found that there a two types of sulfur dioxide -- bounded and free -- in wine.
#Also found that the ratio of free to bounded is considered in the wine making process, so decided to include it 

#Create the training validation split 
from sklearn.cross_validation import train_test_split

#Get dummies for the vintage varibale -- will eliminate the need for imputation as well as the 2001 issue.
dummies_vintage = pd.get_dummies(df.iloc[:,13])

X = pd.concat([dummies_vintage, X], axis=1)

#Get Type variable coded for Red = 0 and White = 1
X.loc[X.type == 'red','type_dummy'] = 0
X.loc[X.type == 'white','type_dummy'] = 1

del X['type']

#Split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.3, random_state = 145)

################# Explantory Model Building #################

import xgboost as xg
from sklearn import tree
from sklearn.linear_model import LogisticRegression
#mxlxtend required a pip install if not already installed on machine
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics import accuracy_score


#### XGBoost for variable selection ####

#Starting here because it requires no missing value imputation 
# --as well as the class imbalance, the xgboost should over sample the'Good' class because it will be harder to classify
#Thus should be able to get an idea for what variables are driving What makes wine 'good' 

#Train the Model
xgb = xg.XGBClassifier(n_estimators=500,
                       seed = 145)
#Increased the n_estimator (number of boosted trees to fit) to 500, to ensure I was getting a good pucture of variable importance
#I wouldn't necessarily spend time tuning this because 

xgb.fit(X_train,Y_train)
xg.plot_importance(xgb)

xg_pred = xgb.predict(X_val)
accuracy = accuracy_score(Y_val,xg_pred, normalize=False)

#The XGBoost has an accuracy rate of .849 on the validation dataset, did this to make sure I wasn't simply capturing variable importances unique to the training data
#Update on all
xgb.fit(X,Y)
xg.plot_importance(xgb)

#Created the same variables and similar results, therefore feel comfortable removing the 2001 vintage and the type variable

#Grab only the important features -- as determined by the xgboost, for use later
X_train_important = X_train.copy()
del X_train_important['type_dummy']
del X_train_important[2001.0]

#### Decision Tree ####

#Impute the NAs with the column medians -- Have to do this for two reasons:
    #1. Python Decision Trees can not handle missing values
    #2. Using the median because of the skewness of the variable distributions
    #If I had more time I would use a more sofisticated imputation method

X_tree_train = X_train_important.copy()

X_tree_train.loc[X_tree_train['volatile acidity'].isnull(), 'volatile acidity'] = X_tree_train.loc[:,'volatile acidity'].median()
X_tree_train.loc[X_tree_train['astringency rating'].isnull(), 'astringency rating'] = X_tree_train.loc[:,'astringency rating'].median()
X_tree_train.loc[X_tree_train['residual sugar'].isnull(), 'residual sugar'] = X_tree_train.loc[:,'residual sugar'].median()
X_tree_train.loc[X_tree_train['pH'].isnull(), 'pH'] = X_tree_train.loc[:,'pH'].median()
#I would not ordinarily impute the residual sugar variable, for a traditional model I would drop it but for a decision tree I will keep it in

X_tree_train[X_tree_train.columns[X_tree_train.isnull().any()]].isnull().sum() #Check to ensure it worked

#Do the same to the validation data
X_tree_val = X_val.copy()
X_tree_val.loc[X_tree_val['volatile acidity'].isnull(), 'volatile acidity'] = X_tree_train.loc[:,'volatile acidity'].median()
X_tree_val.loc[X_tree_val['astringency rating'].isnull(), 'astringency rating'] = X_tree_train.loc[:,'astringency rating'].median()
X_tree_val.loc[X_tree_val['residual sugar'].isnull(), 'residual sugar'] = X_tree_train.loc[:,'residual sugar'].median()
X_tree_val.loc[X_tree_val['pH'].isnull(), 'pH'] = X_tree_train.loc[:,'pH'].median()

del X_tree_val['type_dummy']
del X_tree_val[2001.0]


#Train the Decision Tree
dt = tree.DecisionTreeClassifier(max_depth = 10,
                                 random_state = 145)
dt.fit(X_tree_train,Y_train)


dt_pred = dt.predict(X_tree_val)
accuracy_score(Y_val,dt_pred, normalize=False)

dt_pred_t = dt.predict(X_tree_train)
accuracy_score(Y_train,dt_pred_t, normalize=False)

#If I had more time I would tune the decision tree more.
#When I first ran it to build the tree completely out I had an accruacy is pretty good -- .8267 on validation but it built to far down so I set it up to only build out to 
    # nodes containg at least 5% of the population of the training data, this way it isn't ove fit, the new validation accruacy is .80.
#I would also consider oversampling the response due to the calss imbalance but again the accruacy seems good.
#I would also build out a decision tree for red and one for white to see how different things effect wine quality.

#### Logistic Regression using variable importance from xgboost for relationship information ####
#For this model I will not impute residual sugar I will drop it, and I will look at correlations to avoid multicollinearity problems
#I will also transform my variables -- probably by taking the log with a shift parameter -- to meet the assumptions

X_train_log = X_train_important.copy()

X_train_log.loc[X_train_log['volatile acidity'].isnull(), 'volatile acidity'] = X_train_log.loc[:,'volatile acidity'].median()
X_train_log.loc[X_train_log['astringency rating'].isnull(), 'astringency rating'] = X_train_log.loc[:,'astringency rating'].median()
X_train_log.loc[X_train_log['pH'].isnull(), 'pH'] = X_train_log.loc[:,'pH'].median()

del X_train_log['residual sugar']

import numpy as np

for i in range(6,19):
    X_train_log['log_{}'.format(X_train_log.columns[i])] = np.log(X_train_log[X_train_log.columns[i]]+1)


#Now prep the validation set
X_val_log = X_val.copy()

X_val_log.loc[X_val_log['volatile acidity'].isnull(), 'volatile acidity'] = X_train_log.loc[:,'volatile acidity'].median()
X_val_log.loc[X_val_log['astringency rating'].isnull(), 'astringency rating'] = X_train_log.loc[:,'astringency rating'].median()
X_val_log.loc[X_val_log['pH'].isnull(), 'pH'] = X_train_log.loc[:,'pH'].median()

del X_val_log['residual sugar']
del X_val_log['type_dummy']
del X_val_log[2001.0]

for i in range(6,19):
    X_val_log['log_{}'.format(X_train_log.columns[i])] = np.log(X_val_log[X_val_log.columns[i]]+1)

#Drop originals from both
X_train_log = X_train_log.drop(X_train_log.columns[6:19], axis = 1)
X_val_log = X_val_log.drop(X_val_log.columns[6:19], axis = 1)

#Check Correlations between variables
for i in range(6,19):
    print(X_train_log.columns[i])
    for j in range (6,19):
        print(X_train_log.iloc[:,i].corr(X_train_log.iloc[:,j]),X_train_log.columns[j])
#Fixed Acidity and Astringency Rating are highly correlated -- this makes sense astringency refers to the dryness of the wine and that is usually complemented by sweet or sour
#I will probably choose to include the fixed acidity because it has no imputation
#Also all the sulfur dioxide variables are correlated -- total,free,bounded -- therefore I will only include 1, total since I will capture info in ratio for bounded and free

del X_train_log['log_astringency rating']
del X_val_log['log_astringency rating']
del X_train_log['log_free sulfur dioxide']
del X_val_log['log_free sulfur dioxide']
del X_train_log['log_bounded_so2']
del X_val_log['log_bounded_so2']

#Build the model
lr = LogisticRegression(random_state = 145,
                        solver = 'liblinear')

lr.fit(X_train_log,Y_train)

lr_coef = lr.coef_

lr_pred = lr.predict(X_val_log)
accuracy_score(Y_val,lr_pred, normalize=False)

#If I had more time I would definitely work to improve the accuracy of this model and look at interactions and other things


#### Association Analysis to see if wine quality is associated with certian values of variables ####

## Data Manipulation for this -- Bin my continuous variables: Low(below mean/median), Average(mean/median), and High(above mean/median) ##
df_assoc = df.copy()

for i in range(1,13):
    df_assoc.loc[df_assoc[df_assoc.columns[i]] < df_assoc[df_assoc.columns[i]].mean() ,'low_{}'.format(df_assoc.columns[i])] = 1
    df_assoc.loc[df_assoc[df_assoc.columns[i]] >= df_assoc[df_assoc.columns[i]].mean() ,'low_{}'.format(df_assoc.columns[i])] = 0
    df_assoc.loc[df_assoc[df_assoc.columns[i]] == df_assoc[df_assoc.columns[i]].mean() ,'avg_{}'.format(df_assoc.columns[i])] = 1
    df_assoc.loc[df_assoc[df_assoc.columns[i]] != df_assoc[df_assoc.columns[i]].mean() ,'avg_{}'.format(df_assoc.columns[i])] = 0
    df_assoc.loc[df_assoc[df_assoc.columns[i]] > df_assoc[df_assoc.columns[i]].mean() ,'high_{}'.format(df_assoc.columns[i])] = 1
    df_assoc.loc[df_assoc[df_assoc.columns[i]] <= df_assoc[df_assoc.columns[i]].mean() ,'high_{}'.format(df_assoc.columns[i])] = 0

df_assoc = df_assoc.drop(df_assoc.columns[1:13], axis = 1)

df_assoc = pd.concat([dummies_vintage, df_assoc], axis=1)

del df_assoc['vintage']
del df_assoc['quality']
del df_assoc['quality_cat']

df_assoc.loc[df_assoc.type == 'red','type_dummy'] = 0
df_assoc.loc[df_assoc.type == 'white','type_dummy'] = 1

del df_assoc['type']

df_assoc[df_assoc.isnull()] = 0
#Run the Analysis
r = []
for i in range(1,13):
    frequent_itemsets = apriori(df_assoc[['low_{}'.format(df.columns[i]),'avg_{}'.format(df.columns[i]),'high_{}'.format(df.columns[i]),'quality_bin']], min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    r.append(rules.head())

#Examine the Results

################# Display and Visualize 'What makes a wine 'good'?' #################
#### Update Models on all Data ####
X.loc[X['volatile acidity'].isnull(), 'volatile acidity'] = X.loc[:,'volatile acidity'].median()
X.loc[X['astringency rating'].isnull(), 'astringency rating'] = X.loc[:,'astringency rating'].median()
X.loc[X['residual sugar'].isnull(), 'residual sugar'] = X.loc[:,'residual sugar'].median()
X.loc[X['pH'].isnull(), 'pH'] = X.loc[:,'pH'].median()

del X['type_dummy']
del X[2001.0]

## Decision Tree ##
dt.fit(X,Y)
accuracy_score(dt.predict(X),Y)

Y.value_counts()
dt.predict(X).sum()
## Logistic Regression ##
X_log = X.copy()

del X_log['residual sugar']

for n in range(6,19):
    X_log['log_{}'.format(X_log.columns[n])] = np.log(X_log[X_log.columns[n]]+1)

#Drop originals from both
X_log = X_log.drop(X_log.columns[6:19], axis = 1)

del X_log['log_astringency rating']
del X_log['log_free sulfur dioxide']
del X_log['log_bounded_so2']

lr.fit(X_log,Y)

#### View Results ####
import graphviz 
#This is trickier than simply pip installing the graphviz package -- I had to also download a file with executables in it from https://graphviz.gitlab.io/_pages/Download/Download_windows.html
#Once that is done make the working directory the path to that file.

os.chdir(r'C:\Users\Victoria\Anaconda3\Lib\site-packages\release\bin')
dot_data = tree.export_graphviz(dt, out_file= None , 
                         feature_names=X_tree_train.columns,  
                         class_names=df.quality_cat,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

#Shows a nice flow  of the decision tree and what variables split -- ideally I would output this graph, hard to look at in pannel 

lr_coef = lr.coef_

#Look at coefficients of logistic regression for the purpose of looking at how the important variables effect wine quality

r

#Look at what forms of the variables are most associated with 'good' wine -- not a traditional use of association analysis, infact there is probably a better way to accomplish this but helpful to look at

###If I had more time I would work to output the results better -- prettier
# Quick Anlysis:
    #Good wines tend to have lower values acidity but higher values of citric acid
    #They tend to have lower astringency ratings -- not too dry
    #They tend to have lower levels chlorides, and lower sulfur dioxide levels 
    #They tend to have higher alcohol content and pH

    #Based on the logistic regression model:
        #Alcohol is driving the prediction of a good wine -- largest coefficient -- Want higher value
        #It seems also that higher levels of sulphates increase the odds of a good wine
        #Where as other predictors decrease the odds of having a good wine -- higher chlorides, higher values of volatile acidity
            #These are consistent with the association analysis
            #Some weren't higher pH gave lower odds for a good wine.