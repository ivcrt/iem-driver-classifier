import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv('outputs/iem_driver_dataset.csv')
df = df[df['driver_class'].isin(['DD', 'BA'])]

label_map = {'DD': 0, 'BA': 1}
y = df['driver_class'].map(label_map).values
X = df.drop(columns=['model', 'source', 'driver_class', 'driver_raw']).values
groups = df['model'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67)
train_idx, val_idx = next(gss.split(X, y, groups=groups))

scale_weight = sum(y[train_idx] == 0) / sum(y[train_idx] == 1)

xgb_model = XGBClassifier(
    random_state=1,
    scale_pos_weight=scale_weight,
    eval_metric='logloss'
)

xgb_model.fit(X[train_idx], y[train_idx])
preds = xgb_model.predict(X[val_idx])

print("Accuracy:", accuracy_score(y[val_idx], preds))
print("Macro-F1:", f1_score(y[val_idx], preds, average='macro'))