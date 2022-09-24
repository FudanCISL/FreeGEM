# FreeGEM (NeurIPS 2022)

This is the implementation for our NeurIPS 2022 paper:

> Parameter-free Dynamic Graph Embedding for Link Prediction.
> Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Ning Gu.
> The Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS). 2022.

We provide the code which can be used to reproduce the results of **Table 3: Accuracy comparison with state-of-the-art methods on two link prediction tasks.** 

## Introduction

There are four folders in this project, namely `data`, `preprocess`, `future-item-recommendation` and `next-interaction-prediction`. We will introduce how to use them in turn.

Note that the model reads the processed dataset, so you must first use the script in `preprocess` folder to preprocess the raw data in `data` and obtain the processed dataset before running the model in `future-item-recommendation` and `next-interaction-prediction`.

We welcome you to contact the authors or open issues when you encounter any problems.

## `data`

The raw dataset is saved in this folder, all datasets are public and available.

- For future item recommendation task
  - Video, Game: http://jmcauley.ucsd.edu/data/amazon/links.html
  - ML-100K, ML-1M: https://grouplens.org/datasets/movielens/
- For next interaction prediction task
  - Wikipedia, Lastfm: http://snap.stanford.edu/jodie/

At the same time, this folder is also used to store processed datasets. The processed datasets can be obtained through the script in `preprocess` folder.

## `preprocess`

This folder contains 3 scripts for preprocessing datasets.

- `amazon.ipynb` is used to preprocess Video and game.
- `ml-100k.ipynb` is used to preprocess ML-100K.

- `ml-1m.ipynb` is used to process ML-1M.

## `future-item-recommendation`

After running the script in `preprocess` to obtain the processed dataset:

- To reproduce the results of `FreeGEM *(with attr)` on `ML-100K` in Table 3(a) 

  `python main.py --dataset ml-100k --attr --beta 15 --dim0 1 --dim1 1 --alpha 3`

- To reproduce the results of `FreeGEM *(with attr)` on `ML-1M` in Table 3(a) 

  `python main.py --dataset ml-1m --attr --beta 50 --dim0 4 --dim1 1 --alpha 3`

- To reproduce the results of `FreeGEM *(no attr)` on `Video` in Table 3(a) 

  `python main.py --dataset video --beta 21 --dim0 128`

- To reproduce the results of `FreeGEM *(no attr)` on `Game` in Table 3(a) 

  `python main.py --dataset game --beta 18 --dim0 256`

- To reproduce the results of `FreeGEM *(no attr)` on `ML-100K` in Table 3(a) 

  `python main.py --dataset ml-100k --beta 60 --dim0 1`

- To reproduce the results of `FreeGEM *(no attr)` on `ML-1M` in Table 3(a) 

  `python main.py --dataset ml-1m --beta 60 --dim0 8`

## `next-interaction-prediction`

After running the script in `preprocess` to obtain the processed dataset:

- To reproduce the results of `FreeGEM`  on `Wikipedia` in Table 3(b) 

  `python main.py --dataset wikipedia --beta 35 --dim 512 --offline 35 --lbd 0.8 --p 1 --g 3 --alpha 2`

- To reproduce the results of `FreeGEM`  on `LastFM` in Table 3(b) 

  `python main.py --dataset lastfm --beta 2 --dim 512 --offline 500 --lbd 0.74 --g 1 --p 2  --alpha 5`
