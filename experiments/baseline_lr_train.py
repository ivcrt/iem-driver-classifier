from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

df = pd.read_csv('../outputs/iem_driver_dataset.csv')
df = df[df['driver_class'].isin(['DD', 'BA', 'Hybrid'])]

label_map = {'DD': 0, 'BA': 1, 'Hybrid': 2}
y = df['driver_class'].map(label_map).values
X = df.drop(columns=['model', 'source', 'driver_class', 'driver_raw']).values
groups = df['model'].values
scaler = StandardScaler()


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67)
train_idx, val_idx = next(gss.split(X, y, groups=groups))
X_train_s = scaler.fit_transform(X[train_idx])
X_val_s = scaler.transform(X[val_idx])

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y[train_idx])
preds_lr = lr.predict(X_val_s)

print("LR Accuracy:", accuracy_score(y[val_idx], preds_lr))
print("LR Macro-F1:", f1_score(y[val_idx], preds_lr, average='macro'))