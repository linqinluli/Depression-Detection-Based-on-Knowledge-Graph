from nltk import data
import tqdm
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import collections
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from transformers.models import bert
from bert_extract import Bertone, TextNet


def read_data(dtype, num):
    DATA_PATH = 'datasets/RSDD/'
    train_path = DATA_PATH + 'training'
    test_path = DATA_PATH + 'testing'
    val_path = DATA_PATH + 'validation'

    if dtype == 'train':
        path = train_path
    elif dtype == 'test':
        path = test_path
    elif dtype == 'val':
        path = val_path
    data = []
    print("begin to load data!")
    with open(path) as f:
        for i in tqdm.tqdm(range(num)):
            lines = f.readline()
            line_data = json.loads(lines)
            if line_data[0]['label'] == 'control':
                line_data[0]['label'] = 0
            elif line_data[0]['label'] == 'depression':
                line_data[0]['label'] = 1
            else:
                continue
            data.append(line_data[0])
    print("Loading finished")
    return data


def read_thesaurus(name):
    DATA_PATH = 'thesaurus/' + name + '.txt'
    data = []
    for line in open(DATA_PATH):
        data.append(line.strip('\n').split())

    return data


def score_count(text, word_dict):
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    if text == None:
        return 0

    new_text = pat_letter.sub(' ', text).strip().lower()

    words = new_text.split()

    cnt = collections.Counter(words)

    tar_words = []
    for word_name in word_dict:
        tar_words.append(read_thesaurus(word_name))
    sum_list = [0 for x in range(len(tar_words))]
    for i in range(len(tar_words)):

        for tar in tar_words[i]:

            sum_list[i] = sum_list[i] + cnt[tar[0]] * float(tar[1])
            # print(float(tar[1]))
    # for tar in tar_words:
    #     # if cnt[tar] != 0:
    #     #     print(text)
    #     sum = sum + cnt[tar]
    return sum_list


def feature_extract(data):
    model_path = 'D:/study/Depression KG/Bert/bert_torch/'
    bin_path = model_path + 'bert-base-uncased.bin'
    config_path = model_path + 'config.json'
    vocab_path = model_path + 'vocab.txt'

    textNet = TextNet(code_length=32)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    words_dic = [
        'suicide', 'distracted', 'restless', 'sleepy', 'slow-moving', 'tired',
        'unconfident', 'unenergetic', 'interested', 'negative', 'stressful'
    ]
    res = np.zeros((len(data), len(words_dic) + 1 + 32))
    label = np.zeros((len(data), ), dtype=int)
    print('beging to extract')
    for i in tqdm.tqdm(range(len(data))):

        num = len(data[i]['posts'])
        # begin_time = int(data[i]['posts'][0][0])
        # time_span = int(data[i]['posts'][num-1][0])-begin_time
        max_score = 0
        max_post = ''
        for post in data[i]['posts']:

            # tmp_stamp = int(post[0])
            # time_score = (tmp_stamp-begin_time)/time_span
            sum_list = score_count(post[1], words_dic)

            for j in range(len(sum_list)):
                res[i][j] = res[i][j] + sum_list[j]
            score = np.sum(res[i])
            if score > max_score:
                max_score = score
                max_post = post[1]
        bert_feature = Bertone(textNet, tokenizer, max_post)
        # print(bert_feature)
        res[i][len(words_dic) + 1:] = bert_feature
        # if res[i][13] == bert_feature[1]:
        #     print('fine okkk')
        if data[i]['label'] == 0:
            label[i] = 0
        else:
            label[i] = 1
        res[i][len(words_dic)] = num
        # print(res[i])
    print('finished')
    return res, label


def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc