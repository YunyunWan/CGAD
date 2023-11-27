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

- **SCAN** [^16] is specifically designed to identify clusters, hub nodes, and outliers within a given graph. Here the nodes that belong to identified clusters are considered anomalies. SCAN only uses the structure of the graph as input.

...

[^16]: Reference details (Author, Title, Journal/Conference, Year, etc.) for SCAN.




