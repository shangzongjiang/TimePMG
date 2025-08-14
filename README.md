# <div align="center"> TimePMG: LLM-Based Time Series Forecasting with Period-Aware Multi-Scale Decomposition and Group-Wise Alignment

✨ This repository provides the official implementation of TimePMG: LLM-Based Time Series Forecasting with Period-Aware Multi-Scale Decomposition and Group-Wise Alignment.
# 1 The framework of TimePMG
TimePMG focuses on reprogramming an embedding-visible large language model, e.g., LLaMA and GPT-2, for time series forecasting, while accounting for the adaptive multiscale decomposition and group-wise alignment. TimePMG consists four main parts: **Dominant Period Extraction (DPE) Module**, **Adaptive Multi-Scale Decomposition Module**, **Group-Wise Alignment (GWA) Module**, and **Multi-Scale Forecasting Module**. The framework of TimePMG is shown as follows: 
![framework](https://github.com/shangzongjiang/TimePMG/blob/main/figures/framework.png)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
  
To install all dependencies:
```
pip install -r requirements.txt
```
# 3 Datasets && Description
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/0a758154e0d44de890e3/), then put the downloaded datasets under the folder `./datasets`.

# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
🚀 We provide experiment scripts for demonstration purpose under the folder `./scripts`. 
```
# the default large language model is LLaMA-7B

# long-term forecasting
bash ./scripts/ETTh1.sh

# short-term forecasting
bash ./scripts/M4.sh

# classification
bash ./scripts/EthanolConcentration.sh

# few-shot learning
bash ./scripts/ETTh1.sh

# zero-shot learning
bash ./scripts/m3_m4.sh
bash ./scripts/m4_m3.sh
```
# 5 Main results
The proposed method outperforms other models on most tasks, including [long-term forecasting](./Long-term_Forecasting/README.md), [short-term forecasting](./Short-term_Forecasting/README.md), [classification](./Classification/README.md), [few-shot learning](./Few-shot_Learning/README.md), and [zero-shot learning](./Zero-shot_Learning/README.md).

## 5.1 Long-term forecasting
![long-term forecasting](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-long.png)
## 5.2 Short-term forecasting
![short-term forecasting](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-shot.png)
## 5.3 Few-shot learning
![few-shot learning](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-few.png)
## 5.4 Zero-shot learning
![zero-shot learning](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-zero.png)
