#Case Study-1–Project Code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================
# Load Dataset
# ============================

R = [
0.2, 0.214, 0.228, 0.242, 0.256, 0.27, 0.284, 0.298, 0.313, 0.327,
0.341, 0.355, 0.369, 0.383, 0.397, 0.411, 0.425, 0.439, 0.453, 0.467,
0.481, 0.495, 0.51, 0.524, 0.538, 0.552, 0.566, 0.58, 0.594, 0.608,
0.622, 0.636, 0.65, 0.664, 0.678, 0.692, 0.707, 0.721, 0.735, 0.749,
0.763, 0.777, 0.791, 0.805, 0.819, 0.833, 0.847, 0.861, 0.875, 0.889,
0.95, 0.964, 0.978, 0.992, 1.006, 1.02, 1.034, 1.048, 1.062, 1.076,
1.09, 1.104, 1.118, 1.132, 1.146, 1.16, 1.174, 1.188, 1.202, 1.216,
1.23, 1.244, 1.258, 1.272, 1.286, 1.3, 1.314, 1.328, 1.342, 1.356,
1.37, 1.384, 1.398, 1.409658723, 1.423468599, 1.437278475,
1.451088352, 1.464898228, 1.478708104, 1.49251798, 1.506327857,
1.520137733, 1.533947609, 1.547757485, 1.561567361, 1.575377238,
1.589187114, 1.60299699, 1.616806866, 1.630616743, 1.644426619,
1.658236495, 1.672046371, 1.685856248, 1.699666124, 1.713476,
1.727285876, 1.741095752, 1.754905629, 1.768715505, 1.782525381,
1.796335257, 1.810145134, 1.82395501, 1.837764886, 1.851574762,
1.865384639, 1.879194515, 1.893004391, 1.906814267, 1.920624143,
1.93443402, 1.948243896, 1.962053772, 1.975863648, 1.989673525,
2.008, 2.022, 2.036, 2.05, 2.064, 2.078, 2.092, 2.106, 2.12, 2.134,
2.148, 2.162, 2.176, 2.19, 2.204, 2.218, 2.232, 2.246, 2.26, 2.274,
2.288, 2.302, 2.316, 2.33, 2.344, 2.358, 2.372, 2.386, 2.4, 2.414,
2.426, 2.438, 2.45, 2.462, 2.474, 2.486, 2.498, 2.51, 2.522, 2.534,
2.546, 2.558, 2.57, 2.582, 2.594, 2.606, 2.618, 2.63, 2.642, 2.654,
2.666, 2.678, 2.69, 2.702, 2.714
]
Energy = [
-0.4483, -0.4578, -0.4672, -0.5013, -0.5133, -0.54175, -0.5548,
-0.5687, -0.5746, -0.5859, -0.6515, -0.5908, -0.7177, -0.756,
-0.798, -0.8213, -0.822, -0.823, -0.824, -0.897, -0.835,
-0.8555, -0.8881, -0.897, -0.9055, -0.9788, -0.9854, -0.9955,
-0.9974, -1.0684, -1.0965, -1.1321, -1.1514, -1.2154, -1.2672,
-1.2835, -1.3294, -1.3324, -1.3545, -1.3604, -1.3544, -1.3501,
-1.3489, -1.3467, -1.3378, -1.3301, -1.3289, -1.3145, -1.3079,
-1.3056, -1.2912, -1.2845, -1.2756, -1.2678, -1.2546, -1.2465,
-1.2323, -1.2103, -1.2076, -1.1948, -1.1834, -1.1759, -1.1673,
-1.1532, -1.1468, -1.1364, -1.1245, -1.1145, -1.1043, -0.9993,
-0.9984, -0.9975, -0.9967, -0.9854, -0.9798, -0.9712, -0.9634,
-0.9578, -0.9434, -0.9312, -0.9243, -0.9134, -0.9045, -0.8993,
-0.8931, -0.8845, -0.8746, -0.8712, -0.8634, -0.8601, -0.8589,
-0.8478, -0.8465, -0.8356, -0.8267, -0.8153, -0.8043, -0.7987,
-0.7834, -0.7737, -0.7634, -0.7601, -0.7529, -0.7469, -0.7376,
-0.7287, -0.7165, -0.7043, -0.6908, -0.6854, -0.6756, -0.6643,
-0.6543, -0.6489, -0.6356, -0.62135, -0.6165, -0.5999, -0.5992,
-0.5946, -0.5809, -0.5743, -0.5632, -0.5543, -0.5421, -0.5376,
-0.5267, -0.5165, -0.5112, -0.5034, -0.4987, -0.4856, -0.4765,
-0.4645, -0.4579, -0.4434, -0.4356, -0.4257, -0.4147, -0.4091,
-0.3916, -0.3849, -0.3792, -0.3686, -0.3582, -0.3692, -0.3582,
-0.3471, -0.3386, -0.3298, -0.3197, -0.3026, -0.2976, -0.2864,
-0.2794, -0.2689, -0.2593, -0.2485, -0.2365, -0.2265, -0.2174,
-0.2054, -0.1976, -0.1835, -0.1743, -0.1698, -0.1612, -0.1589,
-0.1497, -0.1395, -0.1243, -0.1154, -0.1054, -0.1009, -0.0934,
-0.0836, -0.0734, -0.0673, -0.0569, -0.0476, -0.0356
]

# Convert to DataFrame
df = pd.DataFrame({'R': R, 'Energy': Energy})

# Check dataset
assert len(df) > 5, "Dataset too small!"
assert not df.isnull().values.any(), "Dataset contains NaN!"

# Features & Labels
X = df[['R']].values
y = df['Energy'].values.reshape(-1, 1)

# ============================
# Scaling
# ============================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ============================
# Train-test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.20,          # <-- better split
    random_state=42
)

# ============================
# Neural Network Model (Optimized)
# ============================

model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ============================
# Early Stopping + Checkpoint
# ============================

es = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

mc = ModelCheckpoint(
    "best_energy_model.h5",
    monitor='val_loss',
    save_best_only=True
)

# ============================
# Train Model
# ============================

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=800,
    verbose=0,
    callbacks=[es, mc]
)

# ============================
# Predictions
# ============================

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

print("\nR2 Score:", r2_score(y_test_real, y_pred))

# ============================
# Plot: Training & Validation Loss
# ============================

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# ============================
# Plot: Actual vs Predicted
# ============================

plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test_real, label="Actual", s=35)
plt.scatter(X_test, y_pred, label="Predicted", s=35)
plt.title("Energy Prediction (Actual vs Predicted)")
plt.xlabel("R")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

# ============================
# Smooth Final PEC Curve
# ============================

R_smooth = np.linspace(min(R), max(R), 500).reshape(-1,1)
R_smooth_scaled = scaler_X.transform(R_smooth)
Energy_smooth_scaled = model.predict(R_smooth_scaled)
Energy_smooth = scaler_y.inverse_transform(Energy_smooth_scaled)

plt.figure(figsize=(7,5))
plt.plot(R_smooth, Energy_smooth, label="NN Predicted Smooth PEC", linewidth=2)
plt.scatter(R, Energy, s=20, alpha=0.5, label="Original Data")
plt.title("Final Smooth Potential Energy Curve")
plt.xlabel("R")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

print("\n✔ Training complete.")
print("✔ Best model saved as: best_energy_model.h5")

