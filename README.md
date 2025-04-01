# TPPM25: Transformer-based Prediction of PM2.5

## Overview
TPPM25 is a state-of-the-art Transformer-based model designed for forecasting PM2.5 concentrations, a critical air quality indicator. The model leverages Transformer neural networks and various data embedding techniques to improve accuracy in predicting air pollution levels over time. 

This repository contains the code implementation of TPPM25, which has been demonstrated to outperform traditional machine learning models such as LSTM, Bi-LSTM, and ensemble deep learning methods.

## Background
Air pollution, particularly fine particulate matter (PM2.5), poses significant health risks and contributes to respiratory and cardiovascular diseases. Traditional forecasting models often struggle with multivariate dependencies and long-term prediction accuracy. TPPM25 addresses these limitations by incorporating attention-based mechanisms that effectively capture both spatial and temporal relations in air pollution data.

## Features
- **Transformer-based Prediction**: Implements self-attention and cross-attention layers to improve predictive accuracy.
- **Multivariate Forecasting**: Incorporates meteorological data such as temperature, humidity, and aerosol optical depth (AOD) to enhance prediction robustness.
- **Long-Term Stability**: Outperforms LSTM and Bi-LSTM in maintaining accuracy over extended forecasting periods.
- **Spatiotemporal Learning**: Utilizes Tobler’s First Law of Geography to account for spatially related air quality changes.
- **Harmonic Analysis**: Uses Fast Fourier Transform (FFT) to analyze periodic PM2.5 patterns for robust forecasting.

## Experimental Findings

TPPM25 was tested against multiple baselines, including:

- **LSTM & Bi-LSTM**: TPPM25 demonstrated improved accuracy in learning temporal dependencies.
- **Linear Regression & Heuristic Models**: TPPM25 significantly outperformed traditional statistical models.
- **Comparison with Zhang et al. (2021)**: TPPM25 improved upon existing deep learning methods in univariate and multivariate PM2.5 prediction.
- **Spatiotemporal Robustness**: Evaluated using datasets from California and Shanghai, showing superior performance in predicting air quality across different regions.

### California AOD Dataset: MSE Evaluation across Models and K Values

![image](https://github.com/user-attachments/assets/c591b1eb-8476-4b81-ba71-50068e18ec9c)
![image](https://github.com/user-attachments/assets/901f62c2-c62a-4851-9ac6-3fb927596f92)

This figure highlights how TPPM25 consistently outperforms LSTM, Bi-LSTM, Linear, and Heuristic models across different K values in terms of MSE and MAE.

### Performance Metrics on California Dataset

![image](https://github.com/user-attachments/assets/fa57fa82-eff0-4d02-8673-f903a0be0f7b)

TPPM25 achieves the lowest MAE, MSE, SDE, and SMAPE across all compared models, including the approach by Zhang et al. (2021), GNN, and traditional linear regression.

## Dataset
- California Aerosol Optical Depth (AOD) and meteorological data.
- PM2.5 data from five Chinese industrial cities (Beijing, Shanghai, Guangzhou, Chengdu, Shenyang).
- Hourly and daily PM2.5 measurements over multiple time periods.

## Model Architecture
The TPPM25 model is structured with:
- **Encoder-Decoder Architecture**: Based on Transformer neural networks.
- **Time2Vec Embeddings**: Enhances temporal learning.
- **Self-Attention & Cross-Attention**: Extracts intra- and inter-feature relationships.
- **Dilated Causal Convolutions**: Expands receptive fields for improved long-term predictions.

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Pytorch-Lightning
- Scikit-learn
- Pandas

### Clone Repository and Setup Environment
```bash
git clone https://github.com/jcl347/TPPM25
cd TPPM25
conda create -n TPPM25 python==3.8
conda activate TPPM25
pip install -r requirements.txt
pip install -e .
