用于关系抽取的注意力引导图卷积网络(Attention Guided Graph Convolutional Networks)
==========

本文/代码介绍了Attention Guided Graph Convolutional graph convolutional networks (AGGCNs) over 大规模句子级关系抽取任务的依赖树(TACRED)。

You can find the paper [here](https://arxiv.org/pdf/1906.07510.pdf)

模型架构概述见下文。

![AGGCN Architecture](fig/Arch.png "AGGCN Architecture")

## Requirements

Our model was trained on GPU Tesla P100-SXM2 of Nvidia DGX.  

- Python 3 (tested on 3.6.8)

- PyTorch (tested on 0.4.1)

- CUDA (tested on 9.0)

- tqdm

- unzip, wget (for downloading only)

我们已经在这个Repo中发布了我们的训练模型和训练日志。你可以在主目录下找到[logs](https://github.com/Cartus/AGGCN_TACRED/blob/master/logs.txt)，在save_models目录下找到训练过的模型。我们发布的模型达到了69.0%的F1 Score，与ACL原始论文中的报告相同。此外，在我们的Arxiv版本中，我们还报告了F1 Score的平均值和std，统计结果是68.2% +- 0.5%，基于5个训练过的模型。随机种子为0、37、47、72和76。

如果你在不同的环境（包括硬件和软件）上运行代码，不能保证模型与我们发布和报告的模型相同。如果您使用默认设置来训练模型，您将在log.txt中得到完全相同的输出。

## Preparation
这段代码要求你能访问TACRED数据集（需要LDC许可证）。一旦你有了TACRED数据，请将JSON文件放在`dataset/tacred`目录下。
首先，下载并解压GloVe向量。

```
chmod +x download.sh; ./download.sh
```
然后准备单词和初始词向量。

```
python3 prepare_vocab.py --data_dir dataset/tacred --vocab_dir dataset/vocab --glove_dir dataset/glove

python3 prepare_vocab_wiki80.py --data_dir dataset/wiki80 --vocab_dir dataset/vocab --glove_dir dataset/glove

```
这将把单词和单词向量以numpy矩阵的形式写到目录`dataset/vocab`中。
  

## Training
要训练AGGCN模型，运行。
```
bash train_aggcn.sh 1
```
模型checkpoint和日志将被保存到`./saved_models/01`。
关于其他参数的使用细节，请参考`train.py`。

## Evaluation
我们的预训练模型被保存在 saved_models/01 这个目录下。要在测试集上运行评估，请运行。

```
python3 eval.py saved_models/01 --dataset test
```
这将默认使用`best_model.pt`文件。使用`--model checkpoint_epoch_10.pt`来指定一个模型checkpoint文件。

## Retrain
重新加载一个预训练好的模型并对其进行微调，运行。

```
python train.py --load --model_file saved_models/01/best_model.pt --optim sgd --lr 0.001
```

## Related Repo
论文使用了DCGCN模型，详细架构请参考TACL19论文[Densely Connected Graph Convolutional Network for Graph-to-Sequence Learning](https://github.com/Cartus/DCGCN)。代码改编自EMNLP18论文[Graph Convolution over Pruned Dependency Trees Improves Relation Extraction](https://nlp.stanford.edu/pubs/zhang2018graph.pdf)。

## Citation

```
@inproceedings{guo2019aggcn,
 author = {Guo, Zhijiang and Zhang, Yan and Lu, Wei},
 booktitle = {Proc. of ACL},
 title = {Attention Guided Graph Convolutional Networks for Relation Extraction},
 year = {2019}
}
```
