from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

df = pd.read_csv('outputs/iem_driver_dataset.csv')
df = df[df['driver_class'].isin(['DD', 'BA', 'Hybrid'])]

label_map = {'DD': 0, 'BA': 1, 'Hybrid': 2}
y = df['driver_class'].map(label_map).values
X = df.drop(columns=['model', 'source', 'driver_class', 'driver_raw']).values
groups = df['model'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=groups))

forest = RandomForestClassifier(random_state=1)
forest.fit(X[train_idx], y[train_idx])
preds = forest.predict(X[val_idx])

print("Accuracy:", accuracy_score(y[val_idx], preds))
print("Macro-F1:", f1_score(y[val_idx], preds, average='macro'))