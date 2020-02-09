#import requirement libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import precision_score, f1_score, recall_score


# Import dataset which is CSV format
pdata = pd.read_csv("bigdata.csv")
print(pdata.info())
print("----------------------")
print(pdata.head())
print("----------------------")
print(pdata.describe())
print("----------------------")


#Cross Validation
# First - split into Train/Test
features = list(pdata.columns.values)
features.remove('parti')
print("----------------------")
print(features)
print("----------------------")
X = pdata[features]
y = pdata['parti']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



# KNN Algorithm
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=2)
nca_pipe = Pipeline([('knn', clf)])
nca_pipe.fit(X_train, y_train)
y_pred = clf.predict([[1,3,1,0,0,1,1,1,0,1,1,1,1,0]])
print(y_pred)
print("KNN result score : ",nca_pipe.score(X_test, y_test))





# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=7, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Decision Tree score : ",clf.score(X_test, y_test))




# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
print("Random Forest score : ",clf.score(X_test, y_test))




# Support Vector Machine
from sklearn import svm

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test)
# get support vectors
clf.support_vectors_
# get indices of support vectors
clf.support_
#get number of support vectors for each class
clf.n_support_
print("SVM result score : ",clf.score(X_test, y_test))







# Gradient Boosting Classifier
from sklearn import ensemble

clf = ensemble.GradientBoostingClassifier(n_estimators=265,
                                          validation_fraction=0.2,
                                          n_iter_no_change=7,
                                          tol=0.01,
                                          random_state=0)
clf = clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
print("Gradient Tree Boosting Classifier score : ",clf.score(X_test, y_test))



# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200, tol=1e-0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
print("Stochastic Gradient Descent score : ",clf.score(X_test, y_test))


# Neural Network
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=10)

clf.fit(X_train, y_train)
print("Neural Network score : ",clf.score(X_test, y_test))
y_pred = clf.predict(X_test)

