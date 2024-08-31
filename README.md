# Predictive and Performance-Oriented Approach for Wind Turbine Gearbox Maintenance

This repository is part of a Ph.D. research project conducted by **Al Oautiq H.S.** under the supervision of **Professor Dr. Pronin S.S.**. The research focuses on enhancing the reliability of wind turbine gearboxes through advanced predictive maintenance strategies, utilizing a combination of machine learning, deep learning, and synthetic data generation techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Methods & Materials](#methods--materials)
   - [Generating Synthetic Vibration Data for a Gearbox in a Wind Turbine](#generating-synthetic-vibration-data-for-a-gearbox-in-a-wind-turbine)
   - [Data Preprocessing and Feature Engineering for Gearbox Fault Prediction](#data-preprocessing-and-feature-engineering-for-gearbox-fault-prediction)
   - [Deep Learning Approaches for Predictive Maintenance in Wind Turbines](#deep-learning-approaches-for-predictive-maintenance-in-wind-turbines)
   - [Enhancing Wind Turbine Gearbox Reliability: A Hybrid Deep Learning Approach for Predictive Maintenance](#enhancing-wind-turbine-gearbox-reliability-a-hybrid-deep-learning-approach-for-predictive-maintenance)
3. [Results & Analysis](#results--analysis)
   - [Machine Learning-Driven Gearbox Fault Detection: A Random Forest Approach](#machine-learning-driven-gearbox-fault-detection-a-random-forest-approach)
   - [Optimized Model for Gearbox Fault Detection: A Random Forest Approach Using Vibration Data](#optimized-model-for-gearbox-fault-detection-a-random-forest-approach-using-vibration-data)
4. [Case Studies & Recommendations](#case-studies--recommendations)
   - [Case Study: Implementation of Predictive Maintenance in an Offshore Wind Farm](#case-study-implementation-of-predictive-maintenance-in-an-offshore-wind-farm)
5. [Software Developed](#software-developed)
6. [Contact Information](#contact-information)


## Introduction

This research aims to improve the reliability and maintenance practices of wind turbine gearboxes, which are critical components in wind energy systems. By leveraging synthetic data, advanced machine learning, and deep learning techniques, the project seeks to enhance predictive maintenance strategies and reduce the operational costs associated with gearbox failures.

## Methods & Materials

### Generating Synthetic Vibration Data for a Gearbox in a Wind Turbine

This phase involved the creation of a synthetic dataset to simulate vibration data from a wind turbine gearbox. The generated data serves as a foundation for testing various fault detection models.

- [WT-Gearbox-Vibration-Data-Generator](https://github.com/alouatiq/WT-Gearbox-Vibration-Data-Generator)

### Data Preprocessing and Feature Engineering for Gearbox Fault Prediction

Effective data preprocessing and feature engineering are crucial steps in building accurate predictive models. This stage focused on preparing the data for machine learning by applying various techniques to extract meaningful features from the vibration data.

- [WT-Gearbox-Fault-Prediction-Data-Preprocessing-and-Feature-Engineering](https://github.com/alouatiq/WT-Gearbox-Fault-Prediction-Data-Preprocessing-and-Feature-Engineering)

### Deep Learning Approaches for Predictive Maintenance in Wind Turbines

Deep learning models were developed to predict potential gearbox failures, offering a sophisticated approach to maintenance scheduling.

- [WT-Gearbox-Fault-Detection-ML-DeepLearningModels](https://github.com/alouatiq/WT-Gearbox-Fault-Detection-ML-DeepLearningModels)

### Enhancing Wind Turbine Gearbox Reliability: A Hybrid Deep Learning Approach for Predictive Maintenance

A hybrid deep learning model was proposed to further enhance the accuracy and reliability of fault detection systems.

## Results & Analysis

### Machine Learning-Driven Gearbox Fault Detection: A Random Forest Approach

This section presents the results of applying a Random Forest classifier to detect gearbox faults, demonstrating the effectiveness of this machine learning technique.

- [WT-Gearbox-Fault-Detection-ML-RandomForestClassifier](https://github.com/alouatiq/WT-Gearbox-Fault-Detection-ML-RandomForestClassifier)

### Optimized Model for Gearbox Fault Detection: A Random Forest Approach Using Vibration Data

Further optimization of the Random Forest model was carried out to improve its performance, using the synthetic vibration data generated earlier.

- [WT-Gearbox-Fault-Prediction-Performance-and-Evaluation-Model](https://github.com/alouatiq/WT-Gearbox-Fault-Prediction-Performance-and-Evaluation-Model)

## Case Studies & Recommendations

### Case Study: Implementation of Predictive Maintenance in an Offshore Wind Farm

The final section of the research applies the developed predictive maintenance models in a real-world scenario, on a wind farm, providing practical insights and recommendations for industry implementation.

## Software Developed

The following software tools were developed during this research:

1. **[WT-Gearbox-Vibration-Data-Generator](https://github.com/alouatiq/WT-Gearbox-Vibration-Data-Generator):** Generates synthetic vibration data for gearbox fault detection.
2. **[WT-Gearbox-Fault-Prediction-Data-Preprocessing-and-Feature-Engineering](https://github.com/alouatiq/WT-Gearbox-Fault-Prediction-Data-Preprocessing-and-Feature-Engineering):** Preprocesses and engineers features from vibration data for predictive modeling.
3. **[WT-Gearbox-Fault-Detection-ML-DeepLearningModels](https://github.com/alouatiq/WT-Gearbox-Fault-Detection-ML-DeepLearningModels):** Implements deep learning models for fault detection.
4. **[WT-Gearbox-Fault-Detection-ML-RandomForestClassifier](https://github.com/alouatiq/WT-Gearbox-Fault-Detection-ML-RandomForestClassifier):** A machine learning model using Random Forest for fault detection.
5. **[WT-Gearbox-Fault-Prediction-Performance-and-Evaluation-Model](https://github.com/alouatiq/WT-Gearbox-Fault-Prediction-Performance-and-Evaluation-Model):** An optimized Random Forest model for fault prediction.

## Stracture
```
Vibration-Based Prediction of Gearbox Faults/
├── Vibration Generator/
│   ├── main.py
│   ├── export_data.py
│   ├── simulation.py
│   └── parameters.py
├── Data Preprocessing and Feature Engineering/
│   ├── data_preprocessing.py
│   └── feature_engineering.py
├── Deep Learning Models/
│   ├── Model_CNN.py
│   ├── Model_LSTM.py
│   ├── Model_Hybrid.py
│   └── Model_Comparison.py
├── Model Performance and Evaluation/
│   ├── FFT_Analysis.py
│   └── Decision_Boundaries.py
├── Random forest classifier Model/
│   ├── hyperparameter_tuning.py
│   └── model_training.py
├── GEN_data/
│   ├── vibration_data.csv
│   ├── Processed_Vibration_Data_Stage_I.csv
│   └── Engineered_Features_Stage_I.csv
├── HIS_data/
│   ├── cnn_history.pkl
│   ├── lstm_history.pkl
│   └── hybrid_history.pkl
├── plots/
│    ├── CNN_Model_Performance
│    ├── LSTM_Model_Performance
│    ├── Hybrid_Model_Performance
│    ├── Model_Comparison
│    └── ...
├── requirements.txt
├── README.md
└── AUTHORS
```

## How to Clone the Repository with Submodules

To properly clone this repository along with its submodules (which represent the associated software projects), use the following command:

```bash
git clone --recurse-submodules https://github.com/alouatiq/Vibration-Based-Prediction-of-Gearbox-Faults.git
```
If you have already cloned the repository without submodules, you can initialize and update the submodules with:

```bash
git submodule update --init --recursive
```

## Contact Information
For further information, discussions, or questions, you can reach out to the author:

**AL OUATIQ H.S.** - [alh@dr.com](mailto:alouatiq@example.com)

**PRONIN S.P.** - [](mailto:)
