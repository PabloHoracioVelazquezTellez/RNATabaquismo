import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Semilla para reproducibilidad
np.random.seed(42)

# Simulación de datos
n_samples = 400

data = {
    'edad': np.random.randint(17, 30, size=n_samples),
    'promedio': np.round(np.random.uniform(70.0, 100.0, size=n_samples), 2),
    'horas_sueno': np.random.randint(3, 11, size=n_samples),
    'estres': np.random.randint(1, 11, size=n_samples),
    'actividad_fisica': np.random.randint(0, 8, size=n_samples),
    'amigos_que_fuman': np.random.randint(0, 11, size=n_samples),
    'ocupacion': np.random.randint(0, 2, size=n_samples),
    'forma_compra': np.random.randint(0, 2, size=n_samples),
    'sexo': np.random.randint(0, 2, size=n_samples),
    'semestre': np.random.randint(1, 13, size=n_samples),
}

# Variable dependiente: cantidad de cigarros por semana
cigarros = (
    2 * data['estres'] +
    1.5 * data['amigos_que_fuman'] -
    0.5 * data['actividad_fisica'] +
    1.2 * data['ocupacion'] +
    0.8 * data['forma_compra'] +
    np.random.normal(0, 5, size=n_samples)  # ruido
)
cigarros = np.clip(cigarros, 0, 100)

# Crear DataFrame
df = pd.DataFrame(data)
df['cigarros_semana'] = cigarros

# Separar entrada y salida
X = df.drop(columns=['cigarros_semana'])
y = df['cigarros_semana']

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # salida
])

model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Entrenamiento
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Evaluación
loss, mae = model.evaluate(X_test, y_test)
print(f"\nPérdida (MSE): {loss:.2f} | MAE: {mae:.2f}")

# Gráfica del entrenamiento
plt.plot(history.history['mae'], label='MAE entrenamiento')
plt.plot(history.history['val_mae'], label='MAE validación')
plt.xlabel('Épocas')
plt.ylabel('Error absoluto medio')
plt.title('Entrenamiento del modelo con datos simulados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
