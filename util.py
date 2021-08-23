from nltk import data
import tqdm
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import collections
import numpy as np


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


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''


def read_thesaurus(name):
    DATA_PATH = 'thesaurus/' + name + '.txt'
    data = []
    for line in open(DATA_PATH):
        data.append(line.strip('\n'))
    data.append(name)
    return data


def word_count(text, word_dict):


    lmtzr = WordNetLemmatizer()
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    if text == None:
        return 0

    new_text = pat_letter.sub(' ', text).strip().lower()
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')

    # finished replace abbreviations

    words = new_text.split()
    # new_words = []
    # for word in words:
    #     if word:
    #         tag = nltk.pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]
    #         pos = get_wordnet_pos(tag[0][1])
    #         if pos:
    #             lemmatized_word = lmtzr.lemmatize(word, pos)
    #             new_words.append(lemmatized_word)
    #         else:
    #             new_words.append(word)
    cnt = collections.Counter(words)

    tar_words = []
    for word_name in word_dict:
        tar_words.append(read_thesaurus(word_name))
    sum_list = [0 for x in range(len(tar_words))]
    for i in range(len(tar_words)):
        for tar in tar_words[i]:
            sum_list[i] = sum_list[i] + cnt[tar]
    # for tar in tar_words:
    #     # if cnt[tar] != 0:
    #     #     print(text)
    #     sum = sum + cnt[tar]
    return sum_list


def feature_extract(data):
    words_dic = [
        'suicide', 'distracted', 'restless', 'sleepy', 'slow-moving', 'tired',
        'unconfident', 'unenergetic'
    ]
    res = np.zeros((len(data), len(words_dic)))
    label = np.zeros((len(data), ), dtype=int)
    print('beging to extract')
    for i in tqdm.tqdm(range(len(data))):

        num = len(data[i]['posts'])
        for post in data[i]['posts']:
            sum_list = word_count(post[1], words_dic)
            for j in range(len(sum_list)):
                res[i][j] = res[i][j] + sum_list[j]
        if data[i]['label'] == 0:
            label[i] = 0
        else:
            label[i] = 1

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