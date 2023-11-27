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

 

## Baselines


