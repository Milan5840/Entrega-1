import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# Cargar datos procesados
data_file = 'data/datos_procesados.csv'
df = pd.read_csv(data_file)

# Separar características y variable objetivo
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
