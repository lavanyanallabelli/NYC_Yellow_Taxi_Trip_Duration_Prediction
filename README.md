
# 🚕 NYC Yellow Taxi Trip Duration Prediction (Jan 2020)

## 🧠 Overview

This project uses **Deep Learning models built in TensorFlow** to predict the duration of yellow taxi rides in New York City. It combines ride data from the **NYC Taxi and Limousine Commission (TLC)** with January 2020 **climatic conditions from the Meteostat portal** to improve prediction accuracy. The aim is to evaluate and compare the performance of three neural network architectures—**Linear Regression, MLP, and Deep Neural Networks (DNN)**—using various loss functions and optimizers.

---

## 📁 Datasets Used

### 1. NYC Yellow Taxi Data (Jan 2020)
- Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Fields used: `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `trip_distance`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, etc.

### 2. NYC Weather Data (Jan 2020)
- Source: [Meteostat – Wall Street, NYC](https://meteostat.net/en/place/3BBKPQ?t=2020-01-01/2020-01-31)
- Fields used: `temperature (°C)`, `precipitation`, `humidity`, `wind speed`, etc.

> The two datasets were merged based on the ride's `pickup_datetime` and the corresponding weather conditions.

---

## 🧹 Data Preprocessing

- **Timestamp Parsing**: Converted to datetime formats.
- **Duration Calculation**: `(dropoff_datetime - pickup_datetime)` in seconds.
- **Feature Engineering**: Extracted time-of-day, day-of-week, and weather-based features.
- **Scaling**: StandardScaler applied to numeric features to normalize the value ranges.
- **Dataset Merge**: Merged taxi rides with weather data on a daily basis.
- **Train/Test Split**: Chronological split (not random!) → 80% training / 20% validation.

---

## 🔧 Model Architectures

### 1. Linear Regression (Baseline)
- Implemented using `tf.keras.Sequential` with no hidden layers.
- Acts as a baseline deep learning model.

### 2. MLP (Multi-Layer Perceptron)
- One hidden layer with 64 neurons and ReLU activation.
- Output layer with a single unit for regression.

### 3. DNN (Deep Neural Network)
- At least 2 hidden layers (e.g., 64 → 32 neurons).
- Regularization added (Dropout or L2) to address overfitting.

---

## ⚙️ Training Configuration

- **Loss Functions**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Optimizers Tested**:
  - SGD (with various learning rates)
  - Adam
  - RMSProp
- **Epochs**: 100
- **Batch Size**: Default (32)
- **Tools**: Trained using `TensorFlow` and optionally visualized with **TensorBoard**
- **Metrics**: Training vs Validation Loss, MAE, and MSE

---

## 📈 Model Evaluation & Visualization

- Plotted `training_loss` vs `validation_loss` for each model
- Reported final MAE and MSE for all 3 models
- Identified the **best model** (typically DNN or MLP with tuned hyperparameters)
- Used `.predict()` to forecast trip durations on the test set

---

## 🏆 Best Model Summary

- **Architecture**: DNN (2 hidden layers with 64 and 32 units)
- **Loss Function**: MAE
- **Optimizer**: Adam (lr=0.001)
- **Performance**:
  - MAE ≈ _xx_ seconds
  - MSE ≈ _xx_ seconds²

*(Fill in exact numbers after running the experiments.)*

---

## 💡 Key Observations

- MLP/DNN outperformed simple linear regression significantly.
- Feature scaling had a notable impact on convergence and stability.
- MAE was more robust than MSE due to outliers in taxi trip durations.
- Overfitting was mitigated via dropout and early stopping.

---

### 🔥 PyTorch Model

- Recreated the best-performing DNN in **PyTorch**
- Maintained architecture and hyperparameters consistent with TensorFlow version
- Compared training speed, loss curves, and final evaluation metrics
- Provided a qualitative comparison between TensorFlow and PyTorch in terms of:
  - Flexibility
  - Training feedback/logging
  - Ease of use

---

## 📦 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/nyc-taxi-duration-dl.git
cd nyc-taxi-duration-dl

# 2. Create virtual environment and activate
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train a model
python train_dnn.py
