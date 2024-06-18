# Graph Bottlenecked Social Recommendation

Overview
--------
Implementation of our KDD'24 accepted paper "Graph Bottlenecked Social Recommendation". 
Our paper is available at: <https://arxiv.org/abs/2406.08214>
<img src="https://github.com/yimutianyang/KDD24-GBSR/blob/main/framework.jpg" width=70%>

In this work, we revisit the general social recommendation and propose a novel Graph Bottlenecked Social Recommendation(GBSR) framework.
Instead of directly taking the observed social networks into formulation, we focus on removing the redundant social relations(noise) to facilitate
recommendation tasks. GBSR is a model-agnostic social denoising framework, that aims to maximize the mutual information between the denoised social graph and recommendation labels, while minimizing it between the denoised social graph and the original one. This enables GBSR to preserve the minimal yet efficient social structure. Technically, GBSR consists of two elaborate components, preference-guided social graph refinement, and HSIC-based bottleneck learning. Experiments conducted on three datasets demonstrate the effectiveness of the proposed GBSR framework.

Prerequisites
-------------
* Please refer to requirements.txt

Usage-Tensorflow
-----
* python run_GBSR.py --dataset douban_book --runid 40+2.5sigma --beta 40 --sigma 0.25
* python run_GBSR.py --dataset yelp --runid 2.0+0.25sigma --beta 2.0 --sigma 0.25
* python run_GBSR.py --dataset epinions --runid 3.0+0.25sigma --beta 3.0 --sigma 0.25

Usage-Pytorch
-----
* cd torch_version
* python run_GBSR.py --dataset douban_book --runid 40+2.5sigma --beta 40 --sigma 0.25
* python run_GBSR.py --dataset yelp --runid 2.0+0.25sigma --beta 2.0 --sigma 0.25
* python run_GBSR.py --dataset epinions --runid 3.0+0.25sigma --beta 3.0 --sigma 0.25

Notice
------
* All experimental results reported in the paper are based on **TensorFlow implementation**.
* There are slight differences in Pytorch implementation. Higher performances on the double_book dataset while lower on other datasets.
* Running speed also differs from platforms, GBSR runs much faster on the Tensorflow platform than Pytorch.

Citation
--------
If you find this useful for your research, please kindly cite the following paper:<be>
```
@article{GBSR2024,
  title={Graph Bottlenecked Social Recommendation},
  author={Yonghui Yang, Le Wu, Zihan Wang, Zhuangzhuang He, Richang Hong, and Meng Wang}
  jconference={30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```


Author contact:
--------------
Email: yyh.hfut@gmail.com
