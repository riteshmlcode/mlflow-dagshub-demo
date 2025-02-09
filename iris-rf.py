import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='k.ritesh2887', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/k.ritesh2887/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the random forest model

max_depth = 10
n_estimators = 100

# Apply mlflow
mlflow.set_experiment('iris-dt')

with mlflow.start_run():
    # Create a random forest classifier
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Log the model
    #mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log the metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")  

    # Log the parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.title('Confusion Matrix');

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf, "random-forest-model")
    mlflow.set_tag("model", 'RandomForest')
    mlflow.set_tag('author', 'xcode')



