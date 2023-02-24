import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from tensorflow.python.keras.metrics import Accuracy, Precision, Recall, AUC



# Read Data
columns = []
for i in range(120):
    columns.append(f'x_{i}')

columns = columns + ['v', 'w']

df = pd.read_csv('data.csv')
df.replace([np.inf], 3.5, inplace=True)

df.columns = columns
df.loc[df['v'] == 0.3, "v"] = 1
df.loc[df['v'] == 0.3, "w"] = 0
df.loc[df['w'] == 0.3, "v"] = 0
df.loc[df['w'] == 0.3, "w"] = 1

print("Raw Data")
print(df.head(5))
print(f"Number of Data Entries: {len(df)}")
print(f"Number of v Entries: {len(df[df['v'] == 1])}")
print(f"Number of w Entries: {len(df[df['w'] == 1])}")


# Split into training and test sets
X = df.drop(['v', 'w'], axis=1)
y = df[['v', 'w']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load all models
rf_model = joblib.load('model_random_forest.joblib')
knn_model = joblib.load('model_knn.joblib')
dt_model = joblib.load('model_decision_tree.joblib')
nn_model = joblib.load('model_neural_network.joblib')

# Get predictions
rf_predictions = rf_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)

# Get metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
nn_accuracy = nn_model.evaluate(X_test, y_test)[1]

rf_precision = precision_score(y_test, rf_predictions, average = 'micro')
knn_precision = precision_score(y_test, knn_predictions, average = 'micro')
dt_precision = precision_score(y_test, dt_predictions, average = 'micro')
nn_precision = tf.keras.metrics.Precision()(y_test, nn_predictions)

rf_recall = recall_score(y_test, rf_predictions, average = 'micro')
knn_recall = recall_score(y_test, knn_predictions, average = 'micro')
dt_recall = recall_score(y_test, dt_predictions, average = 'micro')
nn_recall = tf.keras.metrics.Recall()(y_test, nn_predictions)

rf_f1 = f1_score(y_test, rf_predictions, average = 'micro')
knn_f1 = f1_score(y_test, knn_predictions, average = 'micro')
dt_f1 = f1_score(y_test, dt_predictions, average = 'micro')
nn_f1 = 2 * ((nn_precision * nn_recall) / (nn_precision + nn_recall + 1e-10))

rf_roc = roc_auc_score(y_test, rf_predictions, average = 'micro')
knn_roc = roc_auc_score(y_test, knn_predictions, average = 'micro')
dt_roc = roc_auc_score(y_test, dt_predictions, average = 'micro')
nn_roc = tf.keras.metrics.AUC()(y_test, nn_predictions)

# Save metrics to text file
with open('metrics.txt', 'w') as f:
    f.write(f'Random Forest Accuracy: {rf_accuracy}\n')
    f.write(f'Random Forest Precision: {rf_precision}\n')
    f.write(f'Random Forest Recall: {rf_recall}\n')
    f.write(f'Random Forest F1 Score: {rf_f1}\n')
    f.write(f'Random Forest ROC AUC Score: {rf_roc}\n')
    f.write(f'KNN Accuracy: {knn_accuracy}\n')
    f.write(f'KNN Precision: {knn_precision}\n')
    f.write(f'KNN Recall: {knn_recall}\n')
    f.write(f'KNN F1 Score: {knn_f1}\n')
    f.write(f'KNN ROC AUC Score: {knn_roc}\n')
    f.write(f'Decision Tree Accuracy: {dt_accuracy}\n')
    f.write(f'Decision Tree Precision: {dt_precision}\n')
    f.write(f'Decision Tree Recall: {dt_recall}\n')
    f.write(f'Decision Tree F1 Score: {dt_f1}\n')
    f.write(f'Decision Tree ROC AUC Score: {dt_roc}\n')
    f.write(f'Neural Network Accuracy: {nn_accuracy}\n')
    f.write(f'Neural Network Precision: {nn_precision}\n')
    f.write(f'Neural Network Recall: {nn_recall}\n')
    f.write(f'Neural Network F1 Score: {nn_f1}\n')
    f.write(f'Neural Network ROC AUC Score: {nn_roc}\n')

# Define model names and corresponding metrics
model_names = ['Random Forest', 'KNN', 'Decision Tree', 'Neural Network']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Define scores for each model and metric
accuracy_scores = [rf_accuracy, knn_accuracy, dt_accuracy, nn_accuracy]
precision_scores = [rf_precision, knn_precision, dt_precision, nn_precision]
recall_scores = [rf_recall, knn_recall, dt_recall, nn_recall]
f1_scores = [rf_f1, knn_f1, dt_f1, nn_f1]
roc_auc_scores = [rf_roc, knn_roc, dt_roc, nn_roc]

# Define colors for each metric
colors = [(0.2, 0.4, 0.6, 0.8), (0.4, 0.6, 0.8, 0.8), (0.6, 0.8, 1.0, 0.8), (0.8, 0.6, 0.4, 0.8)]

# Define a list of lists for the scores
scores = [accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores]

# Generate plots
for i, metric in enumerate(metrics):
    # Get the scores for the current metric
    metric_scores = scores[i]

    # Calculate the average score for each model
    model_scores = [np.mean(metric_scores[j::len(model_names)]) for j in range(len(model_names))]

    # Create a bar chart for each metric
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(model_names, model_scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.grid(True)

    # Save the plot as a PNG file
    plt.savefig(metric.lower().replace(' ', '_') + '.png')
    plt.clf()
    plt.close()

# Define a function to plot the confusion matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    conf_matrix = multilabel_confusion_matrix(y_test, y_pred)
    n_classes = y_test.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2 if n_classes > 1 else 1, figsize=(10, 5))
    for i in range(n_classes):
        ax = axes[i] if n_classes > 1 else axes
        sns.heatmap(conf_matrix[i], annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f'Confusion matrix for class {i+1} - {model_name}')
    fig.suptitle(f'{model_name} Confusion Matrix', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('confusion_matrix_' + model_name.lower().replace(' ', '_') + '.png')
    plt.clf()
    plt.close()
   

def plot_confusion_matrix_tf(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_test_classes = tf.argmax(y_test, axis=1)
    conf_matrix = tf.math.confusion_matrix(y_test_classes, y_pred_classes)
    n_classes = y_test.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2 if n_classes > 1 else 1, figsize=(10, 5))
    for i in range(n_classes):
        ax = axes[i] if n_classes > 1 else axes
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f'Confusion matrix for class {i+1} - {model_name}')
    fig.suptitle(f'{model_name} Confusion Matrix', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('confusion_matrix_' + model_name.lower().replace(' ', '_') + '.png')
    plt.clf()
    plt.close()
    

# Plot the confusion matrix for each model
plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
plot_confusion_matrix(knn_model, X_test, y_test, "KNN")
plot_confusion_matrix(dt_model, X_test, y_test, "Decision Tree")
plot_confusion_matrix_tf(nn_model, X_test, y_test, "Neural Network")

# Define a function to plot the learning curve
def plot_learning_curve(model, X, y, model_name):
    if 'keras' in str(type(model)):
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=0,
                            validation_split=0.2)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = history.history['accuracy']
        test_scores = history.history['val_accuracy']
    else:
        X, y = shuffle(X, y, random_state=0)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fig = plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training accuracy')
    plt.plot(train_sizes, test_scores_mean, label='Validation accuracy')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Learning Curve')
    plt.legend(loc='best')
    fig.savefig('learning_curve_' + model_name.lower().replace(' ', '_') + '.png')
    plt.clf()
    plt.close()

    

plot_learning_curve(rf_model, X_train, y_train, "Random Forest")
plot_learning_curve(knn_model, X_train, y_train, "KNN")
plot_learning_curve(dt_model, X_train, y_train, "Decision Tree")
plot_learning_curve(nn_model, X_train, y_train, "Neural Network")