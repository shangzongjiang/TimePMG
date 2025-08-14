# <div align="center"> Multi-Scale Hypergraph Meets LLMs: Aligning Large Language Models for Time Series Analysis

# Datasets && Description
📦 You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/0a758154e0d44de890e3/), then put the downloaded datasets under the folder `./datasets`. The detailed descriptions of the datasets for long-term time series forecasting are shown as follows:
![dataset-statistics](https://github.com/shangzongjiang/TimePMG/blob/main/figures/long-few%20datasets.png)
# Running
🚀 We provide the experiment scripts of TimePMG on all dataset under the folder `./scripts`. You can obtain the full results by running the following command:
```
# Train on ETTh1
bash ./scripts/ETTh1.sh
# Train on ETTh2
bash ./scripts/ETTh2.sh
# Train on ETTm1
bash ./scripts/ETTm1.sh
# Train on ETTm2
bash ./scripts/ETTm2.sh
# Train on Traffic
bash ./scripts/traffic.sh
# Train on Electricity
bash ./scripts/electricity.sh
# Train on Weather
bash ./scripts/weather.sh
```
# Full results of few-shot learning
## Few-shot learning results under 5% training data.
![5full results of few-shot learning](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-5few.png)
## Few-shot learning results under 10% training data.
![10full results of few-shot learning](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-10few.png)
