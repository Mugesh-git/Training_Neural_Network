import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = {
    'Size': np.random.uniform(0.5, 2.5, 1000),  # Planet size (in Earth radii)
    'OrbitPeriod': np.random.uniform(0.5, 10, 1000),  # Orbital period (in Earth years)
    'Temperature': np.random.uniform(100, 1000, 1000),  # Surface temperature (Kelvin)
    'AtmosphericComposition': np.random.uniform(0, 1, 1000),  # Atmospheric composition factor
}

df = pd.DataFrame(data)

growth_stage = np.random.choice([0, 1, 2], size=1000, p=[0.4, 0.4, 0.2])
y = growth_stage

X = df.values

y = tf.keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(8, activation='relu'),  # Hidden layer
    Dense(3, activation='softmax')  # Output layer (3 growth stages)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = np.argmax(y_test, axis=1)
print(f"Predicted classes: {predicted_classes}")
print(f"True classes: {true_classes}")
