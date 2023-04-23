# SememeWSD Synonym (SWSDS)
本次代码为 CICAI2022论文：Chinese Word Sense Embedding with
SememeWSD and Synonym Set在minspore框架下的实现。论文中共两个实验 ，
分别是：寻找合适bert的多义词词义消歧实验，以下称为实验一；借助Tencent AILab 预训练好的
词嵌入生成模型，观察本模型对原有模型优化效果的上下文相似度判别实验，以下称为实验二。详细介绍请见原论文。

## 需要安装的依赖包
* mindspore=1.8
* transformers==2.8.0
* OpenHowNet==0.0.1a8
* gensim
* pyemd
* pandas
* tqdm
* numpy
* easydict
* wheel
* lxml

在运行代码前确保运行如下代码以完成OpenHowNet的安装
```python
import OpenHowNet
OpenHowNet.download()
```
# 用法
* 实验一

实验整体代码放置于SememeWSD_main中。

实验采取论文中实现效果较好的bert_base完成实现，其mindspore版本代码通过官方网站移植过来，下载于文件夹bert中，相应预训练权重参数则下载于bert文件下load文件夹中的final.ckpt。可从[网盘](https://pan.baidu.com/s/1N0Ny-cnmOr7z9cJvC9fSwg?pwd=ygan)处下载


执行run_model_ms.py文件则可实现实验一。

本文实验结果输出多义词及其各词性下预测的准确率。

* 实验二

实验整体代码放置于sememe2中。由于其需要使用实验一的词义消歧模型，因此需要保持与SememeWSD_main文件夹的并列。其aux_files中含模型参数文件，于[网盘](https://pan.baidu.com/s/1N0Ny-cnmOr7z9cJvC9fSwg?pwd=ygan)处下载。

之后执行syn_ms.py文件即可实现实验二。

修改如下代码以切换向量
```python
model = KeyedVectors.load('aux_files/tencent_ailab_zh_d200_word2vec.model')
```






