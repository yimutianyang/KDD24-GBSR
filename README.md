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
## Top-20 Recommendation Performances(Original dataset)
* Douban-Book: Recall@20=0.1694, NDCG@20=0.1523
* Yelp: Recall@20=0.1243, NDCG@20=0.0724
* Epinions: Recall@20=0.0793, NDCG@20=0.0464

## Denoising Results under different degrees of added noises(Semi-synthetic dataset)



