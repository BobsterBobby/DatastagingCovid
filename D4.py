# This gist contains a direct connection to a local PostgreSQL database
# called "suppliers" where the username and password parameters are "postgres"

# This code is adapted from the tutorial hosted below:
# http://www.postgresqltutorial.com/postgresql-python/connect/

import psycopg2
import time
import math
import sklearn

# Establish a connection to the database by creating a cursor object
# The PostgreSQL server must be accessed through the PostgreSQL APP or Terminal Shell

# conn = psycopg2.connect("dbname=suppliers port=5432 user=postgres password=postgres")

# Or:
conn = psycopg2.connect(host="www.eecs.uottawa.ca", port = 15432, database="apose046", user="apose046", password="Campus4711")

# Create a cursor object
cur = conn.cursor()

# A sample query of all data from the "vendors" table in the "suppliers" database
 #print("PostgreSQL server information")
 #print(connection.get_dsn_parameters(), "\n")
#cur.execute("""SELECT * FROM public.special_measures_dimension sm""")
#query_results = cur.fetchall()
#print(query_results)

# Decision Tree 
def decisiontree(queryData):
    startTime = time.time()
    
    col_names = ['reported_date','phu_name','resolved']
    
    # load dataset from database
    pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    endTime = time.time()
    totalTime = endTime - startTime
    #TODO get accuracy
    #TODO get percision
    #TODO get recall


# Gradient Boosting    
def GradientBoosting(queryData):
    querySize = len(queryData)
    startTime = time.time()
    #classifier
    X, y = sklearn.datasets.make_hastie_10_2(random_state=0)
    qSeperator = int(math.floor(querySize*0.70))
    X_train, X_test = X[:qSeperator], X[qSeperator:]
    y_train, y_test = y[:qSeperator], y[qSeperator:]

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    clfScore = clf.score(X_test, y_test)

    endTime = time.time()
    totalTime = endTime - startTime
    #TODO get accuracy
    #TODO get percision
    #TODO get recall


# Random Forest
def RandomForrest(queryData):
    querySize = len(queryData)
    startTime = time.time()
    #classiffier
    X_train, X_test = X[:math.floor(querySize*0.7)], X[math.floor(querySize*0.7):]
    y_train, y_test = y[:math.floor(querySize*0.7)], y[math.floor(querySize*0.7):]

    X, y = sklearn.ensemble.make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
    clf = clf.fit(X, y)
    clf.predict
    
    endTime = time.time()
    totalTime = endTime - startTime
    #TODO get accuracy
    #TODO get percision
    #TODO get recall

# Close the cursor and connection to so the server can allocate
# bandwidth to other requests
cur.close()
conn.close()