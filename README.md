<h1 align='center'>Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series (ACSSM)</h1>
<div align="center">
  <a href="https://bw-park.github.io/" target="_blank">Byoungwoo Park</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://hyungi-lee.github.io/" target="_blank">Hyungi Lee</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://juho-lee.github.io/" target="_blank">Juho Lee</a><sup>1</sup><br>
  <sup>1</sup>KAIST <br>
</div>
<br>
<p align="center">
  <a href="https://openreview.net/forum?id=8zJRon6k5v">
    <img src="https://img.shields.io/badge/ICLR2025-blue" alt="ICLR">
  </a>
</p>

In [Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series](https://openreview.net/forum?id=8zJRon6k5v) (**ACSSM**), we propose a multi-marginal Doob’s transform for irregular time series and introduce an efficient (temporally parallel) variational inference algorithm that leverages stochastic optimal control to approximate it.

## Installation
This code is developed with Python3 and Pytorch. To set up an environment with the required packages,
1. Create a virtual environment, for example:
```
conda create -name acssm python=3.10
conda activate acssm
```
2. Install Pytorch according to the [official instructions](https://pytorch.org/get-started/locally/).
3. Install the requirements:
```
pip install -r requirements.txt
```

## Training and Evaluation
By default, the dataset will be downloaded and processed at the first run.
```
python main.py --problem_name <PROBLEM_NAME>
```
To train and evaluate an ACSSM, use the command above, where
- **PROBLEM_NAME** is the tasks such as **person_activity_classification**, **pendulum_regression**, **ushcn_interpolation**, **ushcn_extrapolation**, **physionet_interpolation** and **physionet_extrapolation**.

You can find more details about other configurations in `main.py`, and default settings for each task are available in `configs`.

# Reference 
If you found our work useful for your research, please consider citing our work.

```
@inproceedings{
park2025amortized,
title={Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series},
author={Byoungwoo Park and Hyungi Lee and Juho Lee},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=8zJRon6k5v}
}
```

# Acknowledgements
Our code builds upon an outstanding open source projects and papers:
* [S5: Simplified State Space Layers for Sequence Modeling](https://openreview.net/forum?id=Ai8Hw3AXqks).
* [Modeling Irregular Time Series with Continuous Recurrent Units](https://proceedings.mlr.press/v162/schirmer22a).
* [Multi-Time Attention Networks for Irregularly Sampled Time Series](https://proceedings.mlr.press/v162/schirmer22a).
* [Transformer Hawkes Process](https://proceedings.mlr.press/v119/zuo20a).