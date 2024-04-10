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
> **Recommendation Performances(Original dataset)**

|Datasets|Recall@10|NDCG@20|Recall@20|NDCG@20|
|:---:|:---:|:---:|:---:|:---:|
|Douban-Book|0.1189|0.1451|0.1694|0.1532|
|Yelp|0.0805|0.0592|0.1243|0.0724|
|Epinions|0.0529|0.0385|0.0793|0.0464|

With the increase of noise degree, we can find that GBSR obtains higher gains(Recall@20) compared with LightGCN-S, which effectively demonstrates the denoising ability of GBSR. (3) We compute average social relation confidences and find that fake relations(0.8692) are significantly lower than original relations(1.0004). Furthermore, GBSR can identify over 90% of fake social relations when $\delta=1.0$. Superior recommendation performances and significant noise discrimination verify the social denoising ability of our proposed GBSR.

> **Denoising Results under different degrees of added noises(Semi-synthetic dataset)**

To better evaluate the denoising ability of GBSR, we add the comparisons on the semi-synthetic datasets. Specifically, we inject a certain percentage $\delta$ of fake social relations to the original social graph and compare GBSR with different degree noise scenarios.
* Yelp dataset:
  
| Methods | $\delta=0$ | $\delta=0.2$ | $\delta=0.5$ |$\delta=1.0$ |$\delta=2.0$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
|LightGCN-S| 0.1126|0.1118 |0.1089|0.1073|0.1029 |
| GBSR   | 0.1243 | 0.1235|0.1213|0.1197|0.1152|
|Improvement|10.40%|10.47%|11.39%|11.56%|11.95%|

> **Running Time Comparisons (s/epoch)+convergence epochs**

|Methods|Douban-Book|Yelp|Epinions|
|:---:|:---:|:---:|:---:|
|LightGCN-S|3.60+508|3.08+262|2.67+560|
|Rule-based|3.47+562|2.78+320|2.35+595|
|ESRF|22.32+36|35.40+68|21.46+52|
|GDMSR|22.19+650|20.95+548|16.22+399|
|GBSR|6.97+53|6.93+90|4.96+95|

> **Sparsity Analysis**

We split all users into three sparsity groups according to the number $K$ of their interacted items:
* Low group: $K\in [0,10)$
* Medium group: $K\in [10, 50)$
* High group: $K\in [50,)$

We compare GBSR and the backbone model performances(NDCG@20) under different sparsity groups:
* Douban-Book dataset:
  
|Methods|Low [0,10)|Medium [10,50)|High [50,)|
|:---:|:---:|:---:|:---:|
|LightGCN-S|0.0322|0.0492|0.0726|
|GBSR|0.0375+(16.46%)|0.0542(+10.16%)|0.0783(+7.85%)|

* Epinions dataset:
  
|Methods|Low [0,10)|Medium [10,50)|High [50,)|
|:---:|:---:|:---:|:---:|
|LightGCN-S|0.0322|0.0492|0.0726|
|GBSR|0.0375+(16.46%)|0.0542(+10.16%)|0.0783(+7.85%)|
