# CGAD

## Requirements:
```
pygod == 0.3.1
dgl == 0.4.1
networkx == 2.6.3
```
## Run the demo
python main.py

## Datasets
| Dataset | Node   | Edge                   | Node features                        | Anomalies      |
|:-------:|:------:|:----------------------:|:------------------------------------:|:--------------:|
| Cora    | paper  | citation relationship  | BoW vectors of paper                 | injected       |
| Reddit  | user/subreddit | post relationship | Vector of post's text               | banned users   |
| Books   | book   | frequent co-purchases | Prices, ratings, number of reviews, etc | tagged books   |
| Disney  | movie  | frequent co-purchases | Prices, ratings, number of reviews, etc | marked movies  |

We provide the inj_cora dataset in the Dataset file, other datasets can be found here.

## Baselines
| Algorithm | Year | Graph-based | Generative learning-based | Contrastive learning-based |
|:---------:|:----:|:------------:|:----------:|:--------------------------:|
|   SCAN    | 2007 |       ✔      |            |                            |
|   MLPAE   | 2014 |              |      ✔     |                            |
| DOMINANT  | 2019 |       ✔      |      ✔     |                            |
|   CoLA    | 2021 |       ✔      |            |             ✔              |
|  ANEMONE  | 2021 |       ✔      |            |             ✔              |
|   CONAD   | 2022 |       ✔      |      ✔     |                            |
|  GRADATE  | 2023 |       ✔      |            |             ✔              |
|   CGAD    | 2023 |       ✔      |            |             ✔              |



