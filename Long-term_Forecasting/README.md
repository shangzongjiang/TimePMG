# <div align="center"> TimePMG: LLM-Based Time Series Forecasting with Period-Aware Multi-Scale Decomposition and Group-Wise Alignment

# Datasets && Description
📦 You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/0a758154e0d44de890e3/), then put the downloaded datasets under the folder `./datasets`. The detailed descriptions of the datasets for long-term time series forecasting are shown as follows:
![dataset-statistics](https://github.com/shangzongjiang/TimePMG/blob/main/figures/long-few%20datasets.png)
# Running
🚀 We provide the experiment scripts of TimePMG on all dataset under the folder `./scripts`. You can obtain the full results by running the following command:
```
# Train on ETTh1
bash ./scripts/TimePMG_ETTh1.sh
# Train on ETTh2
bash ./scripts/TimePMG_ETTh2.sh
# Train on ETTm1
bash ./scripts/TimePMG_ETTm1.sh
# Train on ETTm2
bash ./scripts/TimePMG_ETTm2.sh
# Train on Traffic
bash ./scripts/TimePMG_traffic.sh
# Train on Electricity
bash ./scripts/TimePMG_electricity.sh
# Train on Weather
bash ./scripts/TimePMG_weather.sh
```
# Full results of long-term forecasting
![full results of Long-term forecasting](https://github.com/shangzongjiang/TimePMG/blob/main/figures/full-long.png)
