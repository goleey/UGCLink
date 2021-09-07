"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import re
import string

import nltk
import numpy
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle as pkl
import json
from utils import Constants, read_split_data_foursquare, read_split_data_twitter, clean_str, Constants_yelp
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.corpus import wordnet
from gensim.utils import lemmatize
import sys
import torch
sys.path.append("../")
from config import alp_config
punctuation = '~`!#$%^&*()+-=|\';":/.,?><~·！#￥%……&*（）——+=“：’；、。，？》《{}'
my_stopwords = stopwords.words("english")
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', "RT", "rt", "''", "``" , "\\"]:
    my_stopwords.append(w)
my_stopwords = set(my_stopwords)
class MyDataset(Dataset):
    def __init__(self, dataset_name, word2vec_model_path, texts, labels, max_length_sentences=30, max_length_word=35):

        super(MyDataset, self).__init__()
        if dataset_name == "foursquare" or dataset_name == "twitter":
            self.constant = Constants()
        if dataset_name == "yelp":
            self.constant = Constants_yelp()
        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=word2vec_model_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = self.constant.num_classes

        self.document_encode_all = []
        self.label_all = []
        #processing raw text, including removing stopwords, indexing words.
        for i, raw_text in enumerate(self.texts):
            text = re.sub(r"_", " ", raw_text)
            # print(text)
            text = re.sub(r"@[\w]*", "", text)
            # print(text)
            text = re.sub(r"&amp;|&nbsp;|&quot;", "", text)
            # print(text)
            text = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", text)
            # print(text)

            # stem the word
            # stemizer = PorterStemmer("english")
            print(text)
            document_encode = []
            for word in word_tokenize(text=clean_str(text)):
                # word = stemizer.stem(word)
                if word in self.dict and word not in my_stopwords:
                    document_encode.append(self.dict.index(word))
            print(document_encode)
            # wnl = WordNetLemmatizer()
            # original_document = []
            # document = []
            # document_encode = []
            # for word in word_tokenize(text=clean_str(text)):
            #     original_document.append(word)
            #     pos = get_wordnet_pos(nltk.pos_tag(word_tokenize(word))[0][1])
            #     if pos:
            #         word_ = wnl.lemmatize(word, pos)
            #     else:
            #         word_ = word
            #     if word_ not in my_stopwords and word_ not in punctuation:
            #         if word_ in self.dict:
            #             document.append(word_)
            #             document_encode.append(self.dict.index(word_))
            #         else:
            #             document.append(word_)
            #             document_encode.append(0)
            # print(text)
            # print(original_document)
            # print(document)
            print(document_encode)
            if len(document_encode) < self.max_length_word:
                document_encode.extend([0] * (self.max_length_word - len(document_encode)))
            else:
                document_encode = document_encode[:self.max_length_word]
            document_encode_np = np.array(document_encode)
            if np.all(document_encode_np == 0):
                print(text)
                # exit()
            else:
                self.document_encode_all.append(document_encode)
                self.label_all.append( self.constant.loc_id_map[self.labels[i]])
        self.document_encode_all = numpy.array(self.document_encode_all, np.int64)
        self.label_all = numpy.array(self.label_all, np.int64)
    def __len__(self):
        return len(self.label_all)

    def __getitem__(self, index):
        return "", self.document_encode_all[index], self.label_all[index]
def get_wordnet_pos(tag):
    # print(tag)
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
if __name__ == '__main__':
    # path = "/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/"
    # conf = alp_config(path)
    # texts, labels, _, _ = read_split_data(conf)
    # test = MyDataset("./glove.6B.50d.txt", texts, labels)
    # print (test.__getitem__(index=0))

    text = "Your so close to Dean Heights Beauty Salon. If you want to look your best, make an appointment with one of these lovely ladies. I am bored with the balls. He is playing basketballs."
    # stem the word
    # stoplist = stopwords.words('english') + list(string.punctuation)
    print(text)
    stemmer = SnowballStemmer('english')
    print([word
           if word not in my_stopwords else 0
           for word in word_tokenize(text=clean_str(text))])
    print([stemmer.stem(word)
                               if word not in my_stopwords else 0
                               for word in word_tokenize(text=clean_str(text))])
    # print([
    #         [word if  word not in stoplist  else -1 for word in word_tokenize(text=clean_str(sentences))] for sentences
    #         in
    #         sent_tokenize(text=text)])
    # wnl = WordNetLemmatizer()
    # for word in word_tokenize(text):
    #     # print(nltk.pos_tag(word_tokenize(word))[0][1])
    #     print(word)
    #     # print(wnl.lemmatize(word))
    #     # print(lemmatize(word))
    #     print(wnl.lemmatize(word, get_wordnet_pos(nltk.pos_tag(word_tokenize(word))[0][1])))
