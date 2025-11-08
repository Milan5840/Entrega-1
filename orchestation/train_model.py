import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import os

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Real_Estate_Valuation")

# Cargar datos procesados
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(current_dir, 'data', 'datos_procesados.csv')
df = pd.read_csv(data_file)

# Separar características y variable objetivo
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar una nueva corrida de MLflow
with mlflow.start_run():
    # Crear y entrenar el modelo
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Imprimir métricas
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2 Score: {r2:.2f}')

    # Registrar parámetros en MLflow
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # Registrar métricas en MLflow
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })

    # Registrar el modelo en MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Registrar feature importance
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
