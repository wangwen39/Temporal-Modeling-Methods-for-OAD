# Temporal-Modeling-Methods-for-OAD
This is the code of our paper, "A Comprehensive Study on Temporal Modeling for Online Action Detection"  [Arxiv](https://arxiv.org/abs/2001.07501)    
Codes will be avaliable soon.
![framework](https://github.com/wangwen39/Temporal-Modeling-Methods-for-OAD/blob/master/framework.png)
## Abstract
Online action detection (OAD) is a practical yet challenging task, which has attracted increasing attention in recent years. A typical OAD system mainly consists of three modules: a frame-level feature extractor which is usually based on pre-trained deep Convolutional Neural Networks (CNNs), a temporal modeling module, and an action classifier. Among them, the temporal modeling module is crucial which aggregates discriminative information from historical and current features. Though many temporal modeling methods have been developed for OAD and other topics, their effects are lack of investigation on OAD fairly. This paper aims to provide a comprehensive study on temporal modeling for OAD including four meta types of temporal modeling methods, i.e. temporal pooling, temporal convolution, recurrent neural networks, and temporal attention, and uncover some good practices to produce a state-of-the-art OAD system. Many of them are explored in OAD for the first time, and extensively evaluated with various hyper parameters. Furthermore, based on our comprehensive study, we present several hybrid temporal modeling methods, which outperform the recent state-of-the-art methods with sizable margins on THUMOS-14 and TVSeries.
## Environment
* This code is developed with Pytorch-1.2.0, Python3.7.4
## Citation
If you are using the model provided here in in your research, please considering citing:  
@article{Wang2020ACS,  
  title={A Comprehensive Study on Temporal Modeling for Online Action Detection},  
  author={Wen Wang and Xiaojiang Peng and Yu Qiao and Jian Cheng},   
  journal={ArXiv},   
  year={2020},   
  volume={abs/2001.07501}   
}
