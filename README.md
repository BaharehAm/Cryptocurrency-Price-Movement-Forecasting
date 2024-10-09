# Cryptocurrency market trend Forecasting using Deep Learning

This repository contains data and python source codes for the paper "Cryptocurrency price movement prediction by integrating wavelets with hybrid deep learning systems: Evidence from a large data set and the effect of the Coronavirus disease pandemic".

## Abstract

This study investigates the importance of employing high-quality features to enhance the forecasting performance of deep learning algorithms in cryptocurrency markets. We propose a two-stage prediction framework designed for forecasting the directional movement of cryptocurrency prices. In the first stage, a feature extraction method that integrates Discrete Wavelet Transform (DWT) and Long Short-Term Memory-Autoencoder (LSTM-AE) in parallel is employed to generate high-quality features from historical prices and technical indicators. In the second stage, features derived from DWT and LSTM-AE are concatenated, and the resulting feature set (AE-DWT) is fed into a Gated Recurrent Unit (GRU) to predict the directional movement of cryptocurrency prices. 
Our experimental results on a dataset encompassing 25 cryptocurrencies unveil compelling findings. Our baseline models for comparison include the GRU model trained on raw features without any feature extraction applied. The AE-DWT feature set improves the prediction performance of 23 out of the 25 cryptocurrencies. Both classification performances measured by accuracy and F1-score are improved by around 15% and 18%, respectively, when compared to base cases in which no feature extraction is employed. We evaluate the predictive capability of the proposed method by subjecting it to testing during challenging periods, such as the Coronavirus disease 2019 (COVID-19) pandemic. This time frame is marked by the heightened volatility of cryptocurrencies, which presents an obstacle for forecasting models to make accurate predictions. The results show that our approach is robust to highly volatile periods and can achieve comparable prediction results with those of non-crisis periods characterized by lower volatilities.


## Features

- **Feature Extraction**: Utilizes Discrete Wavelet Transform (DWT) and Long Short-Term Memory-Autoencoder (LSTM-AE) to generate high-quality features.
- **Two-Stage Prediction Framework**: Combines feature extraction with a Gated Recurrent Unit (GRU) for directional price movement prediction.
- **Performance Evaluation**: Compares the proposed model's performance with baseline models and evaluates its robustness during volatile periods like the COVID-19 pandemic.

## Requirements

To run the code and reproduce the results, you will need the following libraries:

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- pywt


## The overall diagram of the proposed model

<img width="468" alt="image" src="https://github.com/user-attachments/assets/5081ab2a-67f3-480a-b73b-08dda6b72838">


## Results

Boxplots of out-of-sample accuracies and F1-scores achieved by GRU using various feature sets:

<img width="468" alt="image" src="https://github.com/user-attachments/assets/310ececd-f252-4bd3-82ec-afd2bfeb161a">


Boxplots of accuracies and F1-scores achieved by the AE-DWT feature set in three periods:

<img width="468" alt="image" src="https://github.com/user-attachments/assets/3bf40702-7f08-4e09-9f4f-ecc886a95d4e">






