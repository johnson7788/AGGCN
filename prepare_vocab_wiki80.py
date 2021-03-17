"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper

# python3 prepare_vocab.py dataset/wiki80 dataset/vocab --glove_dir dataset/glove

def parse_args():
    parser = argparse.ArgumentParser(description='为关系抽取准备vocab')
    parser.add_argument('--data_dir', default='dataset/wiki80',help='数据目录 directory.')
    parser.add_argument('--vocab_dir', default='dataset/vocab',help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='如果指定，那么所有单词都小写')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    #输入文件
    train_file = args.data_dir + '/wiki80_train.txt'
    dev_file = args.data_dir + '/wiki80_val.txt'
    test_file = args.data_dir + '/wiki80_val.txt'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    #输出文件
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("加载文件中。。。")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)
    if args.lower:
        print(f"使用小写，token全部转换成小写的")
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, dev_tokens, test_tokens)]

    # load glove
    print("加载 glove词向量...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("从glove中加载了 {} 个单词的向量".format(len(glove_vocab)))
    
    print("使用训练集构建vocab...")
    v = build_vocab(train_tokens, glove_vocab, args.min_freq)

    print("开始计算 oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        # 所有单词数和所有不在vocab中的单词数
        total, oov = count_oov(d, v)
        print("数据 {} oov单词数{}，所有单词数{}， 占比({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("开始构建 embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print(f"保存生成的单词vocab到文件中{vocab_file}")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    print(f"保存生成的词嵌入到文件中{emb_file}")
    np.save(emb_file, embedding)
    print("完成.")

def load_tokens(filename, remove_entities=False):
    """
    收集所有tokens，去除实体tokens
    :param filename: 收集的文件名字是
    :param remove_entities: 是否移除实体
    :return:
    """
    with open(filename) as infile:
        #保存所有token
        tokens = []
        #保存所有样本
        examples = []
        for line in infile:
            d = json.loads(line)
            examples.append(d)
            ts = d['token']
            if remove_entities:
                ss, se, os, oe = d['h']['pos'][0], d['h']['pos'][1], d['t']['pos'][0], d['t']['pos'][1]
                # 不为实体单词创建vocab, 把实体位置改成PAD代表
                ts[ss:se] = ['<PAD>']*(se-ss)
                ts[os:oe] = ['<PAD>']*(oe-os)
                # 过滤掉PAD的 token
                tokens += list(filter(lambda t: t!='<PAD>', ts))
            else:
                tokens += ts
    print("收集到{}个tokens 从 {} 个样本，文件是{}".format(len(tokens), len(examples), filename))
    return tokens

def build_vocab(tokens, glove_vocab, min_freq):
    """
    用token和glove词中构建vocab。
    :param tokens:  所有token
    :param glove_vocab: glove的单词详细
    :param min_freq: token的最小出现次数，小于这个次数，将不加入vocab中
    :return:
    """
    counter = Counter(t for t in tokens)
    # 如果 min_freq > 0，则使用 min_freq，否则保留所有的glove单词
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # 把实体的token和 特殊的token加入vocab , 特殊token包括['<PAD>', '<UNK>']
    # v = constant.VOCAB_PREFIX + entity_masks() + v
    v = constant.VOCAB_PREFIX + v
    print("构建的vocab为{}个单词，原始的token有 {} 个单词".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    """
    计算oov的单词
    :param tokens: 所有token单词
    :param vocab: 生成的单词表中的单词
    :return:  int， int， 返回所有单词数和所有不在vocab中的单词数
    """
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """
    以列表形式获取所有实体mask。不适用于wiki80，wiki80没有所有实体的类别
    """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

if __name__ == '__main__':
    main()


