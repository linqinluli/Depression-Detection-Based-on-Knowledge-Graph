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
    score_list = [0 for x in range(len(tar_words))]
    for i in range(len(tar_words)):

        for tar in tar_words[i]:

            sum_list[i] = sum_list[i] + cnt[tar[0]]
            score_list[i] = score_list[i] + cnt[tar[0]] * float(tar[1])
            # print(float(tar[1]))
    # for tar in tar_words:
    #     # if cnt[tar] != 0:
    #     #     print(text)
    #     sum = sum + cnt[tar]
    return sum_list, score_list


def feature_extract(data):
    model_path = 'D:/study/Depression KG/Bert/bert_torch/'
    bin_path = model_path + 'bert-base-uncased.bin'
    config_path = model_path + 'config.json'
    vocab_path = model_path + 'vocab.txt'

    textNet = TextNet(code_length=32)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    words_dic = [
        'unenergetic', 'slow-moving', 'restless', 'sleepy', 'tired', 'suicide',
        'stressful', 'distracted', 'interested', 'negative', 'unconfident'
    ]
    res = np.zeros((len(data), len(words_dic) + 1 + 32))
    score_res = np.zeros((len(data), len(words_dic) + 1 + 32))
    label = np.zeros((len(data), ), dtype=int)
    print('beging to extract')
    max_post_list = []
    for i in tqdm.tqdm(range(len(data))):

        num = len(data[i]['posts'])
        # begin_time = int(data[i]['posts'][0][0])
        # time_span = int(data[i]['posts'][num-1][0])-begin_time
        max_score = 0
        max_post = ''
        for post in data[i]['posts']:

            # tmp_stamp = int(post[0])
            # time_score = (tmp_stamp-begin_time)/time_span
            sum_list, score_list = score_count(post[1], words_dic)
            for j in range(len(sum_list)):
                res[i][j] = res[i][j] + sum_list[j]
                score_res[i][j] = score_res[i][j] + score_list[j]
            score = np.sum(score_res[i][:11])
            if score > max_score:
                max_score = score
                max_post = post[1]
        max_post_list.append(max_post)
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
    return res, label, score_res, max_post_list


def predict_one(score, predict, text):
    percent = [[2.25, 5.875, 14.125], [5.3125, 12.75, 29.4375],
               [1.5, 3.5, 9.0], [2.75, 7.375, 16.75], [6.25, 14.125, 36.375],
               [0.0, 2.0, 4.0], [6.75, 14.875, 34.75], [0.0, 0.75, 2.0],
               [13.75, 31.75, 83.875], [0.0, 1.0, 5.0], [1.75, 4.5, 11.0]]
    symptom = [0] * 11
    words_dic = [
        'unenergetic', 'slow-moving', 'restless', 'sleepy', 'tired', 'suicide',
        'stressful', 'distracted', 'interested', 'negative', 'unconfident'
    ]
    for i in range(11):
        if (score[i] < percent[i][0]):
            symptom[i] = 0
        elif score[i] >= percent[i][0] and score[i] < percent[i][1]:
            symptom[i] = 1
        elif score[i] >= percent[i][1] and score[i] < percent[i][2]:
            symptom[i] = 2
        else:
            symptom[i] = 3
    print('The user has a', round(predict * 100, 8),
          '% chance of suffering from depression')
    print('It can be inferred from the performance of the user’s Post that the symptoms that may be included are:')
    print('symptom', '\tdegree')
    for i in range(len(words_dic)):
        print(words_dic[i], '\t', symptom[i])
    print('Among them, the post that best reflects his/her depression is:')
    print(text)


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