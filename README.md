# KDD24-GBSR-Project

Overview
--------
Official code of the submission "Graph Bottlenecked Social Recommendation"(KDD2024 ID:759)

Prerequisites
-------------
* Please refer to requirements.txt

Usage
-----
* python run_GBSR.py --dataset douban_book --runid 40GIB+2.5sigma --alpha 40 --sigma 0.25
* python run_GBSR.py --dataset yelp --runid 2.0GIB+0.25sigma --alpha 2.0 --sigma 0.25
* python run_GBSR.py --dataset epinions --runid 3.0+0.25sigma --alpha 3.0 --sigma 0.25

Experimental Results
--------------------
## Recommendation Performances(Original dataset)
|Datasets|Recall@10|NDCG@20|Recall@20|NDCG@20|
|:---:|:---:|:---:|:---:|:---:|
|Douban-Book|0.1189|0.1451|0.1694|0.1532|
|Yelp|0.0805|0.0592|0.1243|0.0724|
|Epinions|0.0529|0.0385|0.0793|0.0464|

## Denoising Results under different degrees of added noises(Semi-synthetic dataset)
To better evaluate the denoising ability of GBSR, we add the comparisons on the semi-synthetic datasets. Specifically, we inject a certain percentage $\delta$ of fake social relations to the original social graph and compare GBSR with different degree noise scenarios.
* Yelp dataset:
| Methods | $\delta=0$ | $\delta=0.2$ | $\delta=0.5$ |$\delta=1.0$ |$\delta=2.0$ |
| --- | --- | --- |---|---|---|
|**LightGCN-S**| 0.1126|0.1118 |0.1089|0.1073|0.1029 |
| **GBSR**   | 0.1243 | 0.1235|0.1213|0.1197|0.1152|
|**Improvement**|10.40%|10.47%|11.39%|11.56%|11.95%|



