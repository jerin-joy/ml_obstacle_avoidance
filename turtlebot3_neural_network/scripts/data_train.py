import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

# y = pd.concat([df.pop(x) for x in ['v', 'w']], axis=1)
# X = df

# Split into training and test sets
X = df.drop(['v', 'w'], axis=1)
y = df[['v', 'w']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Prepared Data")
print("y_train Dataframe\n")
print(y_train.head(5))

print("X_train Dataframe\n")
print(X_train.head(5))

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Tensorflow Neural Network Classifier
X_tensor = tf.convert_to_tensor(X_train)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_tensor, y_train)

# Save all models
joblib.dump(rf_model, 'model_random_forest.joblib')
joblib.dump(knn_model, 'model_knn.joblib')
joblib.dump(dt_model, 'model_decision_tree.joblib')
joblib.dump(model, 'model_neural_network.joblib')
