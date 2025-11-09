import sys
import os
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Asegúrate de poder importar tu módulo de entrenamiento
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.train_models import X_train, y_train

# Configuración MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random_forest_model")

# Entrenamiento del modelo
model = RandomForestRegressor()
model.fit(X_train, y_train)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="random_forest_model"
    )

print("✅ Modelo entrenado y registrado en MLflow.")


MODEL_PATH = "models:/random_forest_model/latest"
model = mlflow.sklearn.load_model(MODEL_PATH)
print("✅ Modelo cargado para servir en API.")

# Configuración FastAPI
app = FastAPI(title="Real Estate Prediction API")

# Definir la estructura de entrada
class InputData(BaseModel):
    No: int
    X1_transaction_date: float
    X2_house_age: float
    X3_distance_to_nearest_MRT_station: float
    X4_number_of_convenience_stores: float
    X5_latitude: float
    X6_longitude: float

@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        df.columns = [
            "No", 
            "X1 transaction date", 
            "X2 house age", 
            "X3 distance to the nearest MRT station", 
            "X4 number of convenience stores", 
            "X5 latitude", 
            "X6 longitude"
        ]
        prediction = model.predict(df)
        return {"predicted_price": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ejecutar API
if __name__ == "__main__":
    uvicorn.run("predicts:app", host="127.0.0.1", port=8000, reload=True)