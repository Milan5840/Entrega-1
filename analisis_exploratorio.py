# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

# 1. Carga de datos
df = pd.read_csv('Real estate valuation data set.csv', 
                 sep=';',              
                 decimal=',',          
                 thousands=None)       

# 2. Limpieza de datos
# Eliminar duplicados
df = df.drop_duplicates()

# Convertir todos los datos con un mismo tipo
print(df.dtypes)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores nulos
df = df.dropna()
print(df.dtypes)
df.to_csv('datos_limpios.csv', index=False)

# 3. Analisis descriptivo

# Vista general
print(df.head())
print(df.info())

print(df.describe())
print("Media:\n", df.mean(numeric_only=True))
print("Mediana:\n", df.median(numeric_only=True))
print("Moda:\n", df.mode().iloc[0])

# 4. Visualizaciones
sns.histplot(df['Y house price of unit area'], kde=True)
plt.show()

sns.boxplot(x=df['X1 transaction date'], color='lightgreen')
plt.title('Boxplot de X1 transaction date')
plt.show()

sns.kdeplot(df['X1 transaction date'], shade=True, color='coral')
plt.title('Densidad de X1 transaction date')
plt.show()

sns.kdeplot(df['X2 house age'], shade=True, color='coral')
plt.title('Densidad de X2 house age')
plt.show()

sns.kdeplot(df['X3 distance to the nearest MRT station'], shade=True, color='coral')
plt.title('Densidad de X3 distance to the nearest MRT station')
plt.show()

sns.kdeplot(df['X4 number of convenience stores'], shade=True, color='coral')
plt.title('Densidad de X4 number of convenience stores')
plt.show()

# 5. Correlaciones 

corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de correlación entre variables numéricas')
plt.show()

print(df.corr(numeric_only=True)['X1 transaction date'].sort_values(ascending=False))
sns.scatterplot(x='X1 transaction date', y='Y house price of unit area', data=df)
plt.title('Relación entre X1 transaction date y Y house price of unit area')
plt.show()

print(df.corr(numeric_only=True)['X5 latitude'].sort_values(ascending=False))
sns.scatterplot(x='X5 latitude', y='Y house price of unit area', data=df)
plt.title('Relación entre X5 latitude y Y house price of unit area')
plt.show()

print(df.corr(numeric_only=True)['X6 longitude'].sort_values(ascending=False))
sns.scatterplot(x='X6 longitude', y='Y house price of unit area', data=df)
plt.title('Relación entre X6 longitude y Y house price of unit area')
plt.show()

print(df.corr(numeric_only=True)['X3 distance to the nearest MRT station'].sort_values(ascending=False))
sns.scatterplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area', data=df)
plt.title('Relación entre X3 distance to the nearest MRT station y Y house price of unit area')
plt.show()

# 6. Insights adicionales

for col in df.select_dtypes(include='number'):
    print(f"  Columna: {col}")
    print(f"  Media: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Desviación estándar: {df[col].std():.2f}")
    print(f"  Sesgo: {skew(df[col]):.2f}")
    print("-" * 40)
