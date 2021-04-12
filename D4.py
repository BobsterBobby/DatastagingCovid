
# This gist contains a direct connection to a local PostgreSQL database
# called "suppliers" where the username and password parameters are "postgres"

# This code is adapted from the tutorial hosted below:
# http://www.postgresqltutorial.com/postgresql-python/connect/

import psycopg2
import time
import math
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit

# Decision Tree 
def DecisionTree(queryData):

    #classiffier
    y = queryData['resolved']
    X = queryData.drop(['resolved'], axis = 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state=80, stratify = y)
    
    print("\n========== Training Decision Tree ==========\n")
    
    startTime = time.time()
    DT = DecisionTreeClassifier()
    DT = DT.fit(X_train,y_train)
    endTime = time.time()

    # Calculate Math
    totalTime = endTime - startTime
    train_acc = DT.score(X_train,y_train)#get accuracy of train
    test_acc = DT.score(X_test,y_test)#get accuracy of test

    y_pred = DT.predict(X_test)
    precision = precision_score(y_test, y_pred, average = 'macro')#get precision of test
    recall = recall_score(y_test,y_pred, average = 'macro')#get recall of test

    """
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    """
    
    print("Decision Tree Results\n========================")
    print("Training Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Precision Score: ", precision)
    print("Recall Score: ", recall)
    print("Total Elapsed Time in s: ", totalTime)

# Gradient Boosting    
def GradientBoosting(queryData):

    #classiffier
    y = queryData['resolved']
    X = queryData.drop(['resolved'], axis = 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state=0, stratify = y)
    
    print("\n========== Training Gradient Boosting Model ==========\n")
    
    startTime = time.time()
    GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
    #DT = DT.fit(X_train,y_train)
    #randForest.fit(X_train,y_train)
    endTime = time.time()

    # Calculate Math
    totalTime = endTime - startTime
    train_acc = GB.score(X_train,y_train)#get accuracy of train
    test_acc = GB.score(X_test,y_test)#get accuracy of test

    y_pred = GB.predict(X_test)
    precision = precision_score(y_test, y_pred, average = 'macro')#get precision of test
    recall = recall_score(y_test,y_pred, average = 'macro')#get recall of test
    
    print("Gradient Boosting Results\n========================")
    print("Training Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Precision Score: ", precision)
    print("Recall Score: ", recall)
    print("Total Elapsed Time in s: ", totalTime)

# Random Forest
def RandomForest(queryData):
    #classiffier
    y = queryData['resolved']
    X = queryData.drop(['resolved'], axis = 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state=0, stratify = y)

    print("\n========== Training Random Forest ==========\n")
    
    startTime = time.time()
    randForest = RandomForestClassifier()
    randForest.fit(X_train,y_train)
    endTime = time.time()
    
    # Calculate Math
    totalTime = endTime - startTime
    train_acc = randForest.score(X_train,y_train)#get accuracy of train
    test_acc = randForest.score(X_test,y_test)#get accuracy of test

    y_pred = randForest.predict(X_test)
    precision = precision_score(y_test, y_pred, average = 'macro')#get precision of test
    recall = recall_score(y_test,y_pred, average = 'macro')#get recall of test
    
    print("Random Forest Results\n========================")
    print("Training Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Precision Score: ", precision)
    print("Recall Score: ", recall)
    print("Total Elapsed Time in s: ", totalTime)

def main():

    print("\n========== Connecting to Database ==========\n")
    #create connection to database
    credentials = "postgresql://apose046:Campus4711@www.eecs.uottawa.ca:15432/apose046"

    print("\n========== Creating Dataframe ==========\n")
    #create qul query
    dataframe = pd.read_sql("""
                SELECT d.month, d.day, mb.parks, mb.transit_stations, w.daily_low_temp, w.daily_high_temp, w.precipitation, ft.resolved 
                FROM public.covid19_fact_table ft, public.date_dimension d, public.weather_dimension w,
                public.mobility_dimension mb
                WHERE ft.reported_date_id = d.date_id AND ft.weather_id = w.weather_id AND 
                ft.mobility_id = mb.mobility_id
                """, con = credentials)

    # Convert months into number months
    dataframe = dataframe.replace('January', 1)
    dataframe = dataframe.replace('February', 2)
    dataframe = dataframe.replace('March', 3)
    dataframe = dataframe.replace('April', 4)
    dataframe = dataframe.replace('May', 5)
    dataframe = dataframe.replace('June', 6)
    dataframe = dataframe.replace('July', 7)
    dataframe = dataframe.replace('August', 8)
    dataframe = dataframe.replace('September', 9)
    dataframe = dataframe.replace('October', 10)
    dataframe = dataframe.replace('November', 11)
    dataframe = dataframe.replace('December', 12)
    dataframe = dataframe.dropna(how='any',axis=0) #remove NaN
    #We need to get equal number of yes and no

    # print(dataframe.head())

    #start of testing models
    
    print("\n========== Testing Decision Tree Model ==========\n")
    DecisionTree(dataframe)
    print("\n========== Finished Decision Tree Testing ==========\n")

    print("\n========== Testing Gradient Boosting Model ==========\n")
    GradientBoosting(dataframe)
    print("\n========== Finished Gradient Boosting Testing ==========\n")

    print("\n========== Testing Random Forest Model ==========\n")
    RandomForest(dataframe)
    print("\n========== Finished Random Forest Testing ==========\n")

main()
