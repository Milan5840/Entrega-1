import mlflow
import mlflow.sklearn
import pandas as pd
import os
from datetime import datetime

# Configura tu experimento
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Real_Estate_Valuation")

# Ruta del modelo registrado en MLflow
MODEL_PATH = "models:/random_forest_model/1"  # Cambia la versión según tu registro

# Archivo de entrada
INPUT_FILE = "files/datos_procesados.csv"
OUTPUT_DIR = "files/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    try:
        # Cargar modelo desde MLflow
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("Modelo cargado correctamente.")

        # Cargar datos de entrada
        df_input = pd.read_csv(INPUT_FILE)
        print(f"{len(df_input)} registros cargados para predicción.")

        # Validación básica
        if df_input.empty:
            raise ValueError("El archivo de entrada está vacío.")

        # Realizar predicciones
        predictions = model.predict(df_input)
        df_input['predicted_price'] = predictions

        # Guardar resultados con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
        df_input.to_csv(output_file, index=False)
        print(f"Predicciones guardadas en {output_file}")

    except Exception as e:
        print(f"Error en batch predict: {e}")

if __name__ == "__main__":
    main()
