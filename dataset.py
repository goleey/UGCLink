import codecs
import math
import time
from collections import Counter
from pprint import pprint

import nltk
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl
import sys
from utils_ssl import datetime2stamp
from nltk.corpus import stopwords, wordnet
import re, string
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
from bs4 import BeautifulSoup
import scipy.stats as st
from nltk import pos_tag
from nltk.stem import PorterStemmer

punctuation = '~`!#$%^&*()+-=|\';":/.,?><~·！#￥%……&*（）——+=“：’；、。，？》《{}'
my_stopwords = stopwords.words("english")
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', "RT", "rt", ]:
    my_stopwords.append(w)
my_stopwords = set(my_stopwords)
#tf-idf
# word_filtered_dict = set(np.load('/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/word_dict.npy', allow_pickle=True).item().keys())
#lda
# word_filtered_dict = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/common_lda_words.pkl", "rb"))
#document classification-foursquare
word_filtered_dict = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare/document_classification_words.pkl", "rb"))

#document classification-yelp
# word_filtered_dict = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/yelp/document_classification_words.pkl", "rb"))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
#cluster the text in 24 housrs
def _gen_oneday(f_pairs):
    threshold = 3600 * 24.0
    i = 0
    f_oneday = {}
    while i < len(f_pairs):
        if i == len(f_pairs) - 1:
            f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(f_pairs[i][0])
            break
        else:
            jump = 1
            f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(f_pairs[i][0])
            for index_now in range(i + 1, len(f_pairs)):
                # print("index_now", index_now)
                # print("f_pairs", len(f_pairs))
                # print(f_pairs[i][1] - f_pairs[index_now][1])
                if math.fabs(f_pairs[i][1] - f_pairs[index_now][1]) <= threshold:
                    # print(time.strftime('%Y-%m-%d', time.localtime(f_pairs[i][1])))
                    f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(f_pairs[index_now][0])
                    jump += 1
                else:
                    break
            i = i + jump
    return f_oneday
def _filter(f_text, f_time, t_text, t_time, threshold=24 * 3600 * 7.0):
    text_new_pair, time_new = [], []
    f_pairs = list(zip(f_text, f_time))
    t_pairs = list(zip(t_text, t_time))
    sorted(f_pairs, key=lambda x:x[1], reverse=True)
    sorted(t_pairs, key=lambda x:x[1], reverse=True)
    # print(f_pairs)
    # print(t_pairs)
    f_oneday = _gen_oneday(f_pairs)
    t_oneday = _gen_oneday(t_pairs)
    # print(f_oneday)
    # print(t_oneday)
    for f_k, f_v in f_oneday.items():
        for t_k, t_v in t_oneday.items():
            # if f_k == t_k:
            # print(f_k, time.mktime(time.strptime(f_k, "%Y-%m-%d %H:%M:%S")))
            # print(f_v)
            # print(t_k, time.mktime(time.strptime(t_k, "%Y-%m-%d %H:%M:%S")))
            # print(t_v)
            # print(math.fabs(time.mktime(time.strptime(f_k, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(t_k, "%Y-%m-%d %H:%M:%S")))/(3600*24.0))
            # print("----------")
            # print("threshold", threshold/(3600*24))
            if math.fabs(time.mktime(time.strptime(f_k, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(t_k, "%Y-%m-%d %H:%M:%S"))) <= threshold and " ".join(f_v).strip() != "" and " ".join(t_v).strip() != "":
                text_new_pair.append((" ".join(f_v), " ".join(t_v)))
                time_new.append(f_k)
    dic = {}

    for i in range(len(time_new)):
        dic.setdefault(time_new[i], []).append(text_new_pair[i])
    # pprint(dic)

    text_pair = list(dic.values())
    f_agg, t_agg = [], []
    for x in text_pair:
        f_agg.append(x[0][0])
        t_agg.append(" ".join([item[1] for item in x]))
    # print("f_agg", len(f_agg), f_agg)
    # print("t_agg", len(t_agg), t_agg)
    # exit()

    return f_agg, t_agg

class dataset(Dataset):
    def __init__(self, idpair_file, label_file, f_trace, t_trace,threshold=7.0):
        self.datalist = []
        self.idpairs = pkl.load(open(idpair_file, "rb"))
        self.labels = pkl.load(open(label_file, "rb"))
        self.f_trace = pkl.load(open(f_trace, "rb"), encoding="bytes")
        self.t_trace = pkl.load(open(t_trace, "rb"), encoding="bytes")
        self.idpairs_filtered = []
        self.threshold = threshold * 24 * 3600.0
        # len_list = []
        for idx, pair in enumerate(self.idpairs):
            f_user_text = [x[2] for x in self.f_trace[pair[0]]]
            f_user_time = [datetime2stamp(x[1]) for x in self.f_trace[pair[0]]]
            t_user_text = [x[0] for x in self.t_trace[pair[1]]]
            t_user_time = [datetime2stamp(x[1]) for x in self.t_trace[pair[1]]]
            label = self.labels[idx]
            f_user_text, t_user_text = _filter(f_user_text, f_user_time, t_user_text, t_user_time,
                                               threshold=self.threshold)
            text_len = len(t_user_text)
            # len_list.append(text_len)
            if text_len != 0:
                # print(label)
                # print(pair)
                # for index in range(len(f_user_text)):
                #     try:
                #         print("f", f_user_text[index].replace("_", " "))
                #         print("t", t_user_text[index].replace("_", " "))
                #     except Exception:
                #         pass
                # print("---------------------")
                # print("length", len(f_user_text))
                # print("length", len(t_user_text))

                self.datalist.append((pair, f_user_text, t_user_text, text_len, label))
                self.idpairs_filtered.append(pair)

        # d = Counter(len_list)
        # d_s = sorted(d.items(), key=lambda x: x[1], reverse=True)
        # print(d_s)
        # exit()


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.idpairs_filtered[idx], self.datalist[idx]


def collate_fn(batch):
    pair, f_user_text, t_user_text, text_len, labels = [], [], [], [], []
    # text:[bs,user, doc]
    for idpair, x in batch:
        pair.append(x[0])
        f_user_text.append(x[1])
        t_user_text.append(x[2])
        text_len.append(x[3])
        labels.append(x[4])
    return pair, f_user_text, t_user_text, text_len, labels

ps = PorterStemmer()
def _index(doc, word2idx, max_word_length):
    doc = doc.replace("_", " ")
    # stem the word
    stoplist = stopwords.words('english') + list(string.punctuation)
    # stemmer = SnowballStemmer('english')
    doc = re.sub(r"_", " ", doc)
    # print(doc)
    doc = re.sub(r"@[\w]*", "", doc)
    # print(doc)
    doc = re.sub(r"&amp;|&nbsp;|&quot;", "", doc)
    # print(doc)
    doc = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", doc)
    doc = doc.strip().lower()
    doc = re.sub(r"[%s]+" % punctuation, " ", doc)
    # try:
    #     print(doc)
    # except BaseException:
    #     pass
    # x_stem = " ".join(ps.stem(x) for x in word_tokenize(doc))
    # tag = pos_tag(word_tokenize(text=clean_str(x_stem)))
    # filtered_doc_new = [word for word, pos in tag if word in word_filtered_dict
    #                     and word not in stoplist and pos=="NN"]
    # filtered_doc = [word for word in word_tokenize(text=clean_str(doc)) if
    #                 word in word_filtered_dict and word not in stoplist]

    filtered_doc = [word for word in word_tokenize(text=clean_str(doc)) if word not in stoplist]
    # filtered_doc_new = []
    # for x in filtered_doc:
    #     if x != "":
    #         x_stem = ps.stem(x)
    #         pos = pos_tag(word_tokenize(x_stem))
    #         if pos[0][1] == "NN":
    #             filtered_doc_new.append(x_stem)
    filtered_doc_new = filtered_doc
    document_encode = [word2idx[word] for word in filtered_doc_new if word in word2idx]
    # wnl = WordNetLemmatizer()
    # filtered_doc_new = []
    #
    # for word in word_tokenize(text=clean_str(doc)):
    #     pos = get_wordnet_pos(nltk.pos_tag(word_tokenize(word))[0][1])
    #     if pos:
    #         word_ = wnl.lemmatize(word, pos)
    #     else:
    #         word_ = word
    #     if word_ not in stoplist and word_ not in punctuation:
    #         filtered_doc_new.append(word_)
    #     else:
    #         pass
    # document_encode = [word2idx[word] for word in filtered_doc_new if word in word2idx]
    if len(document_encode) < max_word_length:
        document_encode.extend([0] * (max_word_length - len(document_encode)))
    else:
        document_encode = document_encode[:max_word_length]
    document_encode = np.array(document_encode)
    return document_encode, filtered_doc_new, len(filtered_doc_new)


def index_word(pair, f_user_text, t_user_text, text_len, labels, word2idx, max_seq_length, max_word_length_f, max_word_length_t):
    batch_size = len(f_user_text)
    # max_len1 = min(max(f_len), max_seq_length1)
    # max_len2 = min(max(t_len), max_seq_length2)
    # max_len = min(max(text_len), max_seq_length)
    max_len = max_seq_length
    labels = torch.LongTensor(labels)
    f_word_idx = torch.LongTensor(batch_size, max_len, max_word_length_f).fill_(0)
    t_word_idx = torch.LongTensor(batch_size, max_len, max_word_length_t).fill_(0)
    print("==============Next_batch==============")
    for i in range(batch_size):
        if text_len[i] > max_seq_length:
            text_len[i] = max_seq_length

        doc_list_f = []
        doc_list_t = []
        filtered_doc_list_f = []
        filtered_doc_list_t = []
        for idx in range(len(f_user_text[i])):
            doc_f = f_user_text[i][idx]
            doc_idx_f, filtered_doc_f, word_length_f = _index(doc_f, word2idx, max_word_length_f)
            doc_list_f.append(doc_idx_f)

            doc_t = t_user_text[i][idx]
            doc_idx_t, filtered_doc_t, word_length_t = _index(doc_t, word2idx, max_word_length_t)
            doc_list_t.append(doc_idx_t)
            if len(doc_idx_f) !=0 and len(doc_idx_t) != 0:
                filtered_doc_list_f.append(filtered_doc_f)
                filtered_doc_list_t.append(filtered_doc_t)

        text_len[i] = min(len(filtered_doc_list_f), max_seq_length)
        f_word_idx[i, :text_len[i], :] = torch.LongTensor(doc_list_f)[:text_len[i], :]
        t_word_idx[i, :text_len[i], :] = torch.LongTensor(doc_list_t)[:text_len[i], :]
        print("==============next_user==============")
        print(labels[i])
        print(pair[i])
        # for i in range(len(filtered_doc_list_f)):
        #     print("f", filtered_doc_list_f[i])
        #     print("t", filtered_doc_list_t[i])
        # print(f_word_idx.numpy().tolist())
        # print(t_word_idx.numpy().tolist())

    text_len = torch.LongTensor(text_len)

    return f_word_idx, t_word_idx, text_len, labels


def clean_str(string):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # remove punctuation
    try:
        string = BeautifulSoup(string, "lxml").text
    except:
        return ""
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower()
    return s


if __name__ == '__main__':
    # x_train = "/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/x_train"
    # a = pkl.load(open(x_train, "rb"))
    # for x in a:
    #     if x[0] != x[1]:
    #         print(x)

    x_stem = "checking the list before we go out and bring the umbrella. Be a happy girl"
    x_stem = " ".join(ps.stem(x) for x in word_tokenize(x_stem))
    print(x_stem)
    tag = pos_tag(word_tokenize(text=clean_str(x_stem)))
    print(x_stem)

    print([x for x in tag if x[1] == "NN"])
    # if pos[0][1] == "NN":
    #     filtered_doc_new.append(x_stem)