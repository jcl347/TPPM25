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
```

### Example Training Command
```bash
python train.py TPPM25 Data510_Cali_1_k0 --norm layer --target_points 1 --context_points 5 --gpus 0 --batch_size 3 --warmup_steps 1000 --d_model 512 --d_ff 2048 --enc_layers 6 --dec_layers 6 --dropout_emb .10 --dropout_ff .15 --run_name Cali_test_contextextra --base_lr 1e-3 --l2_coeff 1e-3 --loss mse --d_qk 16 --d_v 16 --n_heads 32 --patience 10 --decay_factor .8 --wandb
```

## Logging with Weights and Biases
We used Weights & Biases (wandb) to track all results during development, and you can do the same by providing your username and project as environment variables:

wandb logging can then be enabled with the `--wandb` flag.

There are several figures that can be saved to wandb between epochs. These vary by dataset but can be enabled with `--attn_plot` (for Transformer attention diagrams) and `--plot` (for prediction plotting and image completion).

## Citations
If you use this repository, please cite the following papers:

```bibtex
@article{tong2023robust,
  title={Robust Transformer-based model for spatiotemporal PM2.5 prediction in California},
  author={Weitian Tong and Jordan Limperis and Felix Hamza-Lup and Yao Xu and Lixin Li},
  journal={Earth Science Informatics},
  year={2023},
  doi={10.1007/s12145-023-01138-w}
}

@article{limperis2023pm2.5,
  title={PM2.5 forecasting based on transformer neural network and data embedding},
  author={Jordan Limperis and Weitian Tong and Felix Hamza-Lup and Lixin Li},
  journal={Earth Science Informatics},
  year={2023},
  doi={10.1007/s12145-023-01002-x}
}
```

For further details, refer to the original research papers:
- [PM2.5 Forecasting Based on Transformer Neural Network and Data Embedding](https://doi.org/10.1007/s12145-023-01002-x)
- [Robust Transformer-based Model for Spatiotemporal PM2.5 Prediction in California](https://doi.org/10.1007/s12145-023-01138-w)

