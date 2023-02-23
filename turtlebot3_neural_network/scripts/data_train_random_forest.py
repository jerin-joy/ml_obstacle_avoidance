import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Read Data
columns = []
for i in range(120):
    columns.append(f'x_{i}')
columns = columns + ['v', 'w']
df = pd.read_csv('data.csv')
df.replace([np.inf], 3.5, inplace = True)
df.columns = columns
df.loc[df['v'] == 0.3, "v"] = 1
df.loc[df['v'] == 0.3, "w"] = 0
df.loc[df['w'] == 0.3, "v"] = 0
df.loc[df['w'] == 0.3, "w"] = 1
print("Raw Data")
print(df.head(5))
print(f"Number of Data Entries: {len(df)}")
print(f"Number of v Entries: {len(df[df['v'] == 1])}")
print(f"Number of v Entries: {len(df[df['w'] == 1])}")

y = pd.concat([df.pop(x) for x in ['v', 'w']], axis = 1)
X = df

print("Prepared Data")
print("y Dataframe\n")
print(y.head(5))

print("X Dataframe\n")
print(X.head(5))

# Create Model
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model_rf.fit(X, y)

# Save Model
joblib.dump(model_rf, 'model_rf.joblib')

