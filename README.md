
# Gearbox Fault Detection in Wind Turbines: README

## Project Overview
This project focuses on predictive maintenance for wind turbine gearboxes using machine learning and deep learning techniques. The goal is to accurately detect faults in the gearbox through vibration data analysis. The project includes data preprocessing, feature engineering, machine learning model training, deep learning model implementation, and various analysis scripts.

## Directory Structure
```
├── Data Preprocessing and Feature Engineering/
│   data_preprocessing.py
│   feature_engineering.py
├── Model Performance and Evaluation/
│   Traditional Machine Learning Models/
│   └── Model_Training_+_HyperParameter_Tuning.py
│   Deep Learning Model Training/
│   │── CNN_Model.py
│   │── LSTM_Model.py
│   └── HybridModel.py
├── Model Performance and Evaluation/
│   FFT_Analysis.py
│   Decision_Boundaries.py
├── data/
├── Processed_Vibration_Data_Stage_X.csv
│   └── Engineered_Features_Stage_X.csv
└── README.md
```

## Prerequisites
Make sure you have Python 3.x installed along with the following packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow` (for deep learning models)
- `seaborn` (for visualization)

You can install the necessary packages using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib tensorflow seaborn
```

## Usage Instructions

### 1. Data Preprocessing
Before running any models, the vibration data needs to be preprocessed.
- **Script:** `2-2-1_data_preprocessing.py`
- **Purpose:** Cleans the raw vibration data, handles missing values, and normalizes the data for further analysis.
- **Command:**
  ```bash
  python 2-2-1_data_preprocessing.py
  ```
- **Output:** The processed data is saved as `Processed_Vibration_Data_Stage_X.csv` in the `data/` directory.

### 2. Feature Engineering
Generate additional features from the preprocessed data to improve model performance.
- **Script:** `2-2-2_feature_engineering.py`
- **Purpose:** Extracts statistical features (e.g., mean, standard deviation) from the vibration signals.
- **Command:**
  ```bash
  python 2-2-2_feature_engineering.py
  ```
- **Output:** The engineered features are saved as `Engineered_Features_Stage_I.csv` in the `data/` directory.

### 3. Fast Fourier Transform (FFT) Analysis
Analyze the frequency domain characteristics of the vibration data.
- **Scripts:** `3-3-1_FFT_Analysis.py`
- **Purpose:** Performs FFT on the vibration data to analyze its frequency content.
- **Commands:**
  ```bash
  python 3-3-1_FFT_Analysis.py
  ```
- **Output:** Plots of the FFT results are displayed, showing frequency vs. amplitude for different fault conditions.

### 4. Machine Learning Model Training
Train a Random Forest classifier on the engineered features.
- **Script:** `2-3-1_model_training_+_hyperparameter_tuning.py`
- **Purpose:** Trains a Random Forest model and performs hyperparameter tuning to optimize its performance.
- **Command:**
  ```bash
  python 2-3-1_model_training+hyperparameter_tuning.py
  ```
- **Output:** An optimized Random Forest model is trained, and its performance metrics are displayed.

### 5. Deep Learning Model Training
You can experiment with different deep learning models for fault detection.
- **CNN Model**
  - **Script:** `2-3-2_CNN_Model.py`
  - **Purpose:** Trains a Convolutional Neural Network (CNN) on the vibration data.
  - **Command:**
    ```bash
    python 2-3-2_CNN_Model.py
    ```
- **LSTM Model**
  - **Script:** `2-3-3_LSTM_Model.py`
  - **Purpose:** Trains a Long Short-Term Memory (LSTM) network on the vibration data.
  - **Command:**
    ```bash
    python 2-3-3_LSTM_Model.py
    ```
- **Hybrid Model**
  - **Script:** `2-3-4_HybridModel.py`
  - **Purpose:** Combines a neural network with a Random Forest classifier.
  - **Command:**
    ```bash
    python 2-3-4_HybridModel.py
    ```

### 6. Decision Boundary Visualization
Visualize how well the trained model separates different fault classes.
- **Script:** `3-3-2_Decision_Boundaries.py`
- **Purpose:** Reduces the feature space to two dimensions using PCA and visualizes the decision boundaries of the Random Forest model.
- **Command:**
  ```bash
  python 3-3-2_Decision_Boundaries.py
  ```
- **Output:** A plot showing the decision boundaries of the Random Forest model on the PCA-reduced data.

## Notes and Recommendations
- **Model Selection:** Depending on your data and objectives, you might prefer one model over another. CNNs and LSTMs are powerful but require more computational resources. Random Forests are easier to interpret and faster to train.
- **Feature Engineering:** Good feature engineering can significantly boost the performance of traditional machine learning models like Random Forest.
- **FFT Analysis:** Use FFT analysis if you believe that frequency-domain features are important for detecting faults in your vibration data.
- **Hyperparameter Tuning:** It's important to tune the hyperparameters of your models to achieve the best performance.

## Contact
For any issues or questions, please contact H.S.A.
