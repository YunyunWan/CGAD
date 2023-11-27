# CGAD

## Requirements:
```
pygod == 0.3.1
dgl == 0.4.1
networkx == 2.6.3
```

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
|   CONAD   | 2022 |       ✔      |      ✔     |             ✔              |
|  GRADATE  | 2023 |       ✔      |            |             ✔              |

- **SCAN** [^1] is specifically designed to identify clusters, hub nodes, and outliers within a given graph. Here the nodes that belong to identified clusters are considered anomalies. SCAN only uses the structure of the graph as input. Source code: [https://github.com/pygod-team/pygod](https://github.com/pygod-team/pygod)
- **MLPAE**	[^2] encodes node attributes using a multiple-layer perceptron (MLP), reconstructs the attributes using another MLP, and identifies anomaly nodes based on the reconstruction errors. MLPAE only uses the attribute of the nodes.
- **DOMINANT** [^3] takes both structure and attribute information as input, and employs GCN to reconstruct the structure and attribute separately. The reconstruction error is then used to identify anomalies.
- **CoLA** [^4] measures node abnormality based on the agreement between each node and its neighboring subgraph with a GNN-based encoder model.
- **ANEMONE** [^5] measures node abnormality based on the agreement of multiple scale instance pairs (patch and context levels).
- **CONAD** [^6] enhances the graph based on pre-existing human knowledge and subsequently optimizes the encoder using a contrastive loss function. It then reconstructs the original network to flag anomalies.
- **GRADATE** [^7] adopts random edge modifications as an augmentation strategy and introduces subgraph-subgraph instance pairs into the GAD problem.


## References:
[^1]: X. Xu, N. Yuruk, Z. Feng, and T. A. Schweiger, "Scan: a structural clustering algorithm for networks," in Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining, 2007, pp. 824-833. 
[^2]: M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in Proceedings of the MLSDA 2014 2nd workshop on machine learning for sensory data analysis, 2014, pp. 4-11. 
[^3]: K. Ding, J. Li, R. Bhanushali, and H. Liu, "Deep Anomaly Detection on Attributed Networks," in Proceedings of the 2019 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics., 2019, pp. 594-602.
[^4]: Y. Liu, Z. Li, S. Pan, C. Gong, C. Zhou, and G. Karypis, "Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning," IEEE transactions on neural networks and learning systems, vol. 33, no. 6, pp. 2378-2392, 2021.
[^5]: M. Jin, Y. Liu, Y. Zheng, L. Chi, Y.-F. Li, and S. Pan, "ANEMONE: Graph Anomaly Detection with Multi-Scale Contrastive Learning," in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 2021, pp. 3122-3126. 
[^6]: Z. Xu, X. Huang, Y. Zhao, Y. Dong, and J. Li, "Contrastive attributed network anomaly detection with data augmentation," in Pacific-Asia Conference on Knowledge Discovery and Data Mining 2022, pp. 444-457. 
[^7] J. Duan et al., "Graph anomaly detection via multi-scale contrastive learning networks with augmented view," in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, vol. 37, no. 6, pp. 7459-7467. 



