import os
import pandas as pd
import numpy as np
import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.datasets import fetch_mldata


train = pd.read_csv(r'C:\Users\MyDell\Desktop\CognitiveClass\data_sets\mldata\bsm\train.csv', delimiter=',')
test = pd.read_csv(r'C:\Users\MyDell\Desktop\CognitiveClass\data_sets\mldata\bsm\test.csv', delimiter=',')
validate= pd.read_csv(r'C:\Users\MyDell\Desktop\CognitiveClass\data_sets\mldata\bsm\validation.csv', delimiter=',')

#print(train)
#print(test)
#print(validate)

feature_names = list(train.columns.values)[0:4]
target_names = train['Y'].unique().tolist()
# Formatted data from fitting (training) model
X_trainset = train.drop(train.columns[[0,1]], axis=1).values
y_trainset = train['Y']
# Formated data for test predictions
X_testset = test.drop(test.columns[[0,1]], axis=1).values
y_testset = test['Y']

#print(feature_names)
#print(target_names)
#print(X_trainset)
#print(y_trainset)
#print(X_testset)
#print(y_testset)

#print(X_trainset.shape)
#print(y_trainset.shape)

# Instantiate classifier objects
bsm_neigh8 = KNeighborsClassifier(n_neighbors=8)
bsm_neigh16 = KNeighborsClassifier(n_neighbors=16)
bsm_neigh24 = KNeighborsClassifier(n_neighbors=24)

bsm_forest = RandomForestClassifier(n_estimators=10, criterion='entropy')

# Fit(train) models
bsm_neigh8.fit(X_trainset, y_trainset)
bsm_neigh16.fit(X_trainset, y_trainset)
bsm_neigh24.fit(X_trainset, y_trainset)

bsm_forest.fit(X_trainset, y_trainset)

# Run test predictions, save results
neigh8_pred = bsm_neigh8.predict(X_testset)
neigh16_pred = bsm_neigh16.predict(X_testset)
neigh24_pred = bsm_neigh24.predict(X_testset)

forest_pred = bsm_forest.predict(X_testset)

# Use sklearn.metrics to score models
print(metrics.accuracy_score(y_testset, neigh8_pred))
print(metrics.accuracy_score(y_testset, neigh16_pred))
print(metrics.accuracy_score(y_testset, neigh24_pred))

print(metrics.accuracy_score(y_testset, forest_pred)) # !!!WOW!!! 1.0 means this model was 100% accurate in tests

""" 
    It looks like the Random Forest was 100% accurate in test -- we will focus on using this model in visualization 

    Here, we use a for loop to generate graphs of each decision tree in this instance of the bsm_forest RandomForestClassifier

"""

# Visualize Model

# print any tree 0 with tree index
#print(bsm_forest.estimators_[0]) 

for tree in bsm_forest.estimators_:
    dotfile = export_graphviz(tree,
                             out_file=None,
                             filled=True, rounded=True,  
                             special_characters=True)

    graph = graphviz.Source(dotfile)
    graph.render("tree_graph{}".format(bsm_forest.estimators_.index(tree)))
    

