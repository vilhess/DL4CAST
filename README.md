# Deep Learning for Time Series Forecasting (DL4CAST)

In this repository, we implement several deep learning-based forecasting models. 

---

## Models

The models we consider are:
- [**PatchTST**](https://arxiv.org/abs/2211.14730) - [GitHub: PatchTST](https://github.com/yuqinie98/PatchTST)
- [**MOMENT**](https://arxiv.org/pdf/2402.03885) - [GitHub: MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
- [**TimeMixer**](https://openreview.net/pdf?id=7oLshfEIC2) - [GitHub: TimeMixer](https://github.com/kwuking/TimeMixer)
- [**GPT4TS**](https://arxiv.org/pdf/2302.11939) - [GitHub: GPT4TS](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/tree/main)
- [**iTransformer**](https://arxiv.org/pdf/2310.06625) - [GitHub: iTransformer](https://github.com/thuml/iTransformer)
- [**SAMFormer**](https://arxiv.org/pdf/2402.10198) - [GitHub: SAMFormer](https://github.com/romilbert/samformer)
- [**TOTO**](https://arxiv.org/pdf/2505.14766) - [GitHub: TOTO](https://github.com/DataDog/toto)
- [**TimeXLSTM**](https://arxiv.org/pdf/2405.04517) - [GitHub: xLSTM](https://github.com/NX-AI/xlstm)


When available, we slightly edited the model's code from the original GitHub repository to make it work on our project. We follow the PyTorch Lightning framework.

--- 

## 📊 Datasets

The datasets we consider are:
- **ETT**: Electricity Transformer (ETT) dataset, which includes measurements from two electric transformers (labeled 1 and 2), with two temporal resolutions, 15 minutes (denoted as ‘m’) and 1 hour (denoted as ‘h’). The dataset is available at [ETT Dataset](https://github.com/zhouhaoyi/ETDataset)

- **Weather**: Weather dataset, which includes temperature, humidity, and pressure data. The dataset is available at [Weather Dataset](https://www.bgc-jena.mpg.de/wetter/)

- **Exchange Rate**: Exchange rate dataset, which includes the exchange rates of several currencies. The dataset is available at [Exchange Rate Dataset](https://github.com/laiguokun/multivariate-time-series-data)

- **National Illness**: National illness dataset, which includes weekly records of patient counts and illness ratios. The dataset is available at [National Illness Dataset](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)


---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets

To install the datasets, you can use the following command:
```bash
python preprocess.py
```

### Training

To train a given model on a specified dataset, use the following command:

```python 
python main.py dataset=<dataset_name> model=<model_name> 
``` 

where `<dataset_name>` and `<model_name>` can be one of the following:  


| Models       | Datasets               | 
|-------------|------------------------|
| `toto`     | `etth1`            |
| `gpt4ts`   | `etth2` |
| `itransformer` | `ettm1`                 |  
| `samformer`        | `ettm2`                |  
| `moment`      | `weather`                 |  
| `timemixer`  | `exchange_rate`                |  
| `timexlstm`       | `national_illness`         | 
| `vaformer `   |                        | 
| `patchtst`   |                        | 

---

### Testing 

During testing phase, we evaluate model's performance using MSE and MAE Losses.
Results are saved to:  
```bash
results/
```
### Configurations

For each dataset and each model, the configurations can be view and edit in the ```conf/``` directory, following the [hydra](https://hydra.cc/) framework.
