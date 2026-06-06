#will try adding cross validation later

#to prevent the horrid tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#imports
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report

from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('outputs/iem_driver_dataset.csv')

df = df[df['driver_class'].isin(['DD', 'BA'])] # we'll keep only the most represented ones because the others are under-represented

label_map = {'DD': 0, 'BA': 1}
y = df['driver_class'].map(label_map).values

X = df.drop(columns=['model', 'source', 'driver_class', 'driver_raw']).values
groups = df['model'].values 


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67) #we need to do that because the same iem has measurements ~ 
#  ~ from different people so we dont want to risk having the same iem in training and in validation 
train_idx, val_idx = next(gss.split(X, y, groups=groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]



num_augmentations = 2  # we want to artificially increase our dataset size to 3 times its original size
list_X = [X_train] # lists that will stock the augmented values
list_y = [y_train]

"""
test to see what's the maximum shift we can apply

# First curve of the dataset
curve = X[0] 

# Créons des versions décalées
curve_shift_1 = np.roll(curve, 1)
curve_shift_2= np.roll(curve, 2)
curve_shift_5 = np.roll(curve, 5)


plt.plot(curve, label='Original', linewidth=2, color='black')
plt.plot(curve_shift_1, label='Shift 1 bin ', linestyle='--', color='blue')
plt.plot(curve_shift_2, label='Shift 2 bin ', linestyle='.', color='green')
plt.plot(curve_shift_5, label='Shift 5 bins ', linestyle=':', color='red')
plt.legend()
plt.title("Impact of the bin shift")
plt.show()
"""
# so we will use a shift of 1 bins max (=% of the total length of the curve)

for _ in range(num_augmentations):
    X_aug = X_train.copy()
    shifts = np.random.choice([-1, 1], size=len(X_train))
    for i in range(len(X_aug)):
        X_aug[i] = np.roll(X_aug[i], shift=shifts[i])
    noise = np.random.normal(loc=0.0, scale=0.1, size=X_aug.shape) # 0.1 because an std of 0.1 dB is accepted in the audio world
    X_aug = X_aug + noise
    list_X.append(X_aug)
    list_y.append(y_train)

X_train = np.concatenate(list_X, axis=0)
y_train = np.concatenate(list_y, axis=0)
print(f"Train after augmentation: {X_train.shape} | val : {X_val.shape}")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(X_train.shape, X_val.shape)
print('train:', Counter(y_train))
print('val:  ', Counter(y_val))



model = keras.Sequential([

    keras.Input(shape=(120,)),
    layers.Reshape((120, 1)),
    layers.Conv1D(filters=16, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),                        
    layers.Activation('relu'), 
    layers.Dropout(0.3),  
    layers.Dense(1, activation='sigmoid'), 

])

custom_optimizer = keras.optimizers.Adam(learning_rate=0.0001) # the graph had big jumps so we need to reduce adams speed (normally 0.001)
model.compile(
    optimizer=custom_optimizer,
    loss='binary_crossentropy', 
    metrics=['binary_accuracy'], 
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

class_weights = compute_class_weight(
                                class_weight = "balanced", #calculates weights inversely proportionnal to the class's frequency
                                classes = np.unique(y_train),
                                y = y_train                                                    
                            )
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=300,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
   
)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(title="Loss")
history_df[['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
            history_df['val_binary_accuracy'].max()))


y_probs = model.predict(X_val)
y_pred = (y_probs > 0.5).astype(int).flatten()
print("Macro-F1:", f1_score(y_val, y_pred, average='macro'))
print(classification_report(y_val, y_pred, target_names=['DD', 'BA']))
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['DD', 'BA'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — MLP")
plt.show()