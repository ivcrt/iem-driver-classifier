#will try adding cross validation later


from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/iem_driver_dataset.csv')

df = df[df['driver_class'].isin(['DD', 'BA', 'Hybrid'])] # we'll keep only the most represented ones because the others are under-represented

label_map = {'DD': 0, 'BA': 1, 'Hybrid': 2}
y = df['driver_class'].map(label_map).values

X = df.drop(columns=['model', 'source', 'driver_class', 'driver_raw']).values # so only the frequency response curves

print(X.shape)  
print(y.shape)  
groups = df['model'].values  #we need it for groupshufflesplit


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67) #we need to do that because the same iem has measurements ~ 
#  ~ from different people so we dont want to risk having the same iem in training and in validation 

train_idx, val_idx = next(gss.split(X, y, groups=groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)  


print(X_train.shape, X_val.shape)
print('train:', Counter(y_train))
print('val:  ', Counter(y_val))



model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[120]),
    layers.Dense(64, activation='relu'),    
    layers.Dense(3, activation='softmax'), # because 3 types of drivers so we cant use a sigmoid
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # because 3 categories again
    metrics=['sparse_categorical_accuracy'], # same its not binary anymore (Im comparing to the Kaggle course I took)
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.01,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=300,
    callbacks=[early_stopping],
   
)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(title="Loss")
history_df[['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot(title="Accuracy")
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_sparse_categorical_accuracy'].max())) # <-- Correction ici
plt.show()