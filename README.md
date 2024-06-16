# Graph Bottlenecked Social Recommendation

Overview
--------
Implementation of our KDD'24 accepted paper "Graph Bottlenecked Social Recommendation".
![](https://github.com/yimutianyang/KDD24-GBSR/blob/main/framework.jpg)

In this work, we revisit the general social recommendation and propose a novel Graph Bottlenecked Social Recommendation(GBSR) framework.
Instead of directly taking the observed social networks into formulation, we focus on removing the redundant social relations(noise) to facilitate
recommendation tasks. GBSR is a model-agnostic social denoising framework, that aims to maximize the mutual information between the denoised social graph and recommendation labels, while minimizing it between the denoised social graph and the original one. This enables GBSR to preserve the minimal yet efficient social structure. Technically, GBSR consists of two elaborate components, preference-guided social graph refinement, and HSIC-based bottleneck learning. Experiments conducted on three datasets demonstrate the effectiveness of the proposed GBSR framework.

Prerequisites
-------------
* Please refer to requirements.txt

Usage
-----
* python run_GBSR.py --dataset douban_book --runid 40+2.5sigma --beta 40 --sigma 0.25
* python run_GBSR.py --dataset yelp --runid 2.0+0.25sigma --beta 2.0 --sigma 0.25
* python run_GBSR.py --dataset epinions --runid 3.0+0.25sigma --beta 3.0 --sigma 0.25


Author contact:
--------------
Email: yyh.hfut@gmail.com
