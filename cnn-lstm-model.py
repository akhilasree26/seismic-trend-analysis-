# ==============================
# Import Required Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


# ==============================
# Custom Callback to Print Accuracy
# ==============================

class PrintAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")


# ==============================
# Load Dataset
# ==============================

data = pd.read_csv('/kaggle/input/earthquake-database/database.csv')


# ==============================
# Data Preprocessing
# ==============================

# Remove columns with more than 66% missing values
null_columns = data.columns[data.isna().sum() > 0.66 * len(data)]
data.drop(columns=null_columns, inplace=True, errors='ignore')

# Handle missing values
data['Root Mean Square'] = data['Root Mean Square'].fillna(data['Root Mean Square'].mean())
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Merge Date and Time
data['time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce', utc=True)


# ==============================
# Feature Selection
# ==============================

features = [
    'Latitude',
    'Longitude',
    'Depth',
    'Azimuthal Gap',
    'Horizontal Distance',
    'Horizontal Error',
    'Root Mean Square'
]

data = data[features + ['Magnitude']].copy()

# Handle missing values
data.fillna(0, inplace=True)


# ==============================
# Create Target Variable
# ==============================

data['Target'] = (data['Magnitude'] >= 6.0).astype(int)

X = data.drop(['Target', 'Magnitude'], axis=1).values
y = data['Target'].values


# ==============================
# Handle Imbalanced Data (Oversampling)
# ==============================

minority_idx = np.where(y == 1)[0]
majority_idx = np.where(y == 0)[0]

np.random.seed(42)

oversampled_minority = np.random.choice(minority_idx, size=len(majority_idx), replace=True)

X_balanced = np.concatenate([X[majority_idx], X[oversampled_minority]])
y_balanced = np.concatenate([y[majority_idx], y[oversampled_minority]])


# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)


# ==============================
# Reshape for CNN + LSTM
# ==============================

timesteps = 5
feature_count = X_scaled.shape[1]

n_samples = (X_scaled.shape[0] // timesteps) * timesteps

X_scaled = X_scaled[:n_samples]
y_balanced = y_balanced[:n_samples]

X_reshaped = X_scaled.reshape((-1, timesteps, feature_count))
y_reshaped = y_balanced[::timesteps]


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y_reshaped,
    test_size=0.2,
    random_state=42
)


# ==============================
# Compute Class Weights
# ==============================

weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(weights))


# ==============================
# Build CNN + LSTM Model
# ==============================

model = Sequential([

    Conv1D(512, kernel_size=3, activation='relu', padding='same',
           input_shape=(timesteps, feature_count)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    LSTM(256, return_sequences=True),
    BatchNormalization(),

    LSTM(256, return_sequences=True),
    BatchNormalization(),

    LSTM(128),
    Dropout(0.2),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])


# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ==============================
# Callbacks
# ==============================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print_accuracy = PrintAccuracyCallback()


# ==============================
# Train Model
# ==============================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr, print_accuracy],
    verbose=0
)


# ==============================
# Model Evaluation
# ==============================

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
r2 = r2_score(y_test, y_pred_prob)

print("\nFinal Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("RMSE:", rmse)
print("R2:", r2)


# ==============================
# Visualization
# ==============================

plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# Save Model
# ==============================

model.save("seismic_hybrid_model.h5")
