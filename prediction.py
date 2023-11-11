import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# gets the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# the names for the columns that will be used
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# read the dataset to pandas
iris_df = pd.read_csv(url, header=None, names=columns)

# X contains the features, y contains the labels (class)
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['class']

# data split in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# random seed
model = RandomForestClassifier(random_state=10)

# model training
model.fit(X_train, y_train)

# predictions on the test
y_pred = model.predict(X_test)

# accuracy-evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ser input for feature values
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# predictions based on the input
user_input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
predicted_class = model.predict(user_input_features)

#prints the predicted class of the flower based on the input
print(f"The predicted class is: {predicted_class[0]}")