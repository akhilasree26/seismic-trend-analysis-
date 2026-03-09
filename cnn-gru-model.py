# ==============================
# Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


# ==============================
# Custom Callback
# ==============================

class PrintAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Train Acc = {logs['accuracy']:.4f}, Val Acc = {logs['val_accuracy']:.4f}")


# ==============================
# Load Dataset
# ==============================

data = pd.read_csv('/kaggle/input/earthquake-database/database.csv')

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
data.fillna(0, inplace=True)


# ==============================
# Create Target Variable
# ==============================

data['Target'] = (data['Magnitude'] >= 6.0).astype(int)

X = data.drop(['Target', 'Magnitude'], axis=1).values
y = data['Target'].values


# ==============================
# Balance Dataset (Oversampling)
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
# Reshape Data for CNN + GRU
# ==============================

timesteps = 5
feature_count = X_scaled.shape[1]

n_samples = (X_scaled.shape[0] // timesteps) * timesteps

X_seq = X_scaled[:n_samples].reshape((-1, timesteps, feature_count))
y_seq = y_balanced[:n_samples][::timesteps]


# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq,
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
# Build CNN + GRU Model
# ==============================

model = Sequential([

    Conv1D(256, kernel_size=3, activation='relu', padding='same',
           input_shape=(timesteps, feature_count)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.25),

    GRU(128),
    BatchNormalization(),
    Dropout(0.35),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.25),

    Dense(1, activation='sigmoid')
])


# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==============================
# Training Callbacks
# ==============================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.3,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

print_callback = PrintAccuracyCallback()


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
    callbacks=[early_stop, reduce_lr, print_callback],
    verbose=0
)


# ==============================
# Model Evaluation
# ==============================

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_test = y_test.flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
r2 = r2_score(y_test, y_pred_prob)

print("\nModel Performance")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R² Score  : {r2:.4f}")


# ==============================
# Accuracy Graph
# ==============================

plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy (CNN + GRU)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# Loss Graph
# ==============================

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss (CNN + GRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ==============================
# Probability Distribution
# ==============================

plt.figure(figsize=(10,6))
plt.hist(y_pred_prob[y_test==0], bins=20, alpha=0.7, label='Weak Quake')
plt.hist(y_pred_prob[y_test==1], bins=20, alpha=0.7, label='Strong Quake')
plt.title('Predicted Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# Save Model
# ==============================

model.save('/kaggle/working/cnn_gru_magnitude_model.h5')

print("Model saved successfully")