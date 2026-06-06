import pandas as pd
from tensorflow import keras
import joblib

def predict_driver(csv_path):
    model = keras.models.load_model('saved_models/best_iem_cnn_model.keras')
    scaler = joblib.load('saved_models/scaler.pkl') # needs 120 frequency columns (values)
    df_new = pd.read_csv(csv_path)
    X_new = df_new.values 
    X_new_scaled = scaler.transform(X_new) # use the scaler to prepare the data
    probs = model.predict(X_new_scaled)
    preds = (probs > 0.5).astype(int).flatten()
    label_map_inverse = {0: 'DD', 1: 'BA'}
    results = [label_map_inverse[p] for p in preds]
    print("Predicitions :", results)

if __name__ == "__main__":
    print("Inference test of the CNN : ")
    file= "file.csv" # put your own file name there
    predict_driver(file)

  