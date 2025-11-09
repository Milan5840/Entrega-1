from prefect import flow, task

from src.data.analisis_exploratorio import load_and_process_data
from src.models.train_models import train_and_evaluate

@task
def get_data():
    return load_and_process_data("files/datos_procesados.csv")

@task
def train_models(X_train, y_train, X_test, y_test):
    return train_and_evaluate(X_train, y_train, X_test, y_test)

@flow(name="ML Orchestration Flow")
def ml_pipeline():
    X_train, X_test, y_train, y_test = get_data()
    model, rmse, mae, r2 = train_models(X_train, y_train, X_test, y_test)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    ml_pipeline()
