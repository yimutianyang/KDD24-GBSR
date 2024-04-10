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



