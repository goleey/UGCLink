"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import random
import re
import traceback

import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np
import pickle as pkl
import json
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from torch.utils.data.dataloader import default_collate
class Constants():
    def __init__(self):
        self.training_ratio = 0.8
        self.loc_id_map = {"Arts & Entertainment": 0,
                      "College & University": 1,
                      "Event": 2,
                      "Food": 3,
                      "Nightlife Spot": 4,
                      "Outdoors & Recreation": 5,
                      "Professional & Other Places": 6,
                      "Residence": 7,
                      "Shop & Service": 8,
                      "Travel & Transport": 9}
        self.loc_list = ["Arts & Entertainment",
                      "College & University",
                      "Event",
                      "Food",
                      "Nightlife Spot",
                      "Outdoors & Recreation",
                      "Professional & Other Places",
                      "Residence",
                      "Shop & Service",
                      "Travel & Transport"]
        self.num_classes = len(self.loc_id_map)
        self.max_word_length, self.max_sent_length = 140, 5
        self.threshold = 0.5
class Constants_yelp():
    def __init__(self):
        self.training_ratio = 0.8
        self.loc_id_map = {}
        self.loc_list = ["Active Life", "Arts & Entertainment", "Automotive", "Beauty & Spas", "Education", "Event Planning & Services", "Financial Services", "Food", "Health & Medical", "Home Services", "Hotels & Travel", "Local Flavor", "Local Services", "Mass Media", "Nightlife", "Pets", "Professional Services", "Public Services & Government", "Real Estate", "Religious Organizations", "Restaurants", "Shopping"]
        for k, v in enumerate(self.loc_list):
            self.loc_id_map[v] = k
        self.num_classes = len(self.loc_id_map)
        self.max_word_length, self.max_sent_length = 140, 5
        self.threshold = 0.5
constant = Constants()
def clean_str(string):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # remove punctuation
    string = BeautifulSoup(string, "lxml").text
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
    s =string.strip().lower()
    return s
def my_collate_fn(batch):
    batch = list(filter(lambda x: not np.all(x == 0), batch))
    return default_collate(batch)
def read_split_data_foursquare(config):
    foursquare_checkins = pkl.load(open(config.foursquare_checkins, "rb"), encoding="bytes")
    loc2category = json.load(open(config.loc2category_foursquare))
    constant = Constants()
    texts = []
    labels = []
    for k, v in list(foursquare_checkins.items()):
        for item in v:
            loc = item[0]
            text = item[2]
            if loc in loc2category:
                loc_list = loc2category[loc]
                for loc_name in loc_list:
                    texts.append(text)
                    labels.append(loc_name)
    x_train = texts[:int(len(texts) * constant.training_ratio)]
    y_train = labels[:int(len(texts) * constant.training_ratio)]
    x_test = texts[int(len(texts) * constant.training_ratio):]
    y_test = labels[int(len(texts) * constant.training_ratio):]
    return x_train, y_train, x_test, y_test
def read_split_data_twitter(config):
    lines = open(config.twitter_poi,encoding='utf-8').readlines()
    twitter_poi = []
    for line in lines:
        json_str = line.strip()
        json_dict = json.loads(json_str)
        twitter_poi.extend(json_dict["content"])

    loc2category = json.load(open(config.loc2category_foursquare))
    constant = Constants()
    texts = []
    labels = []
    for item in twitter_poi:
        loc = item[0]
        text = item[1]
        if loc in loc2category:
            loc_list = loc2category[loc]
            for loc_name in loc_list:
                texts.append(text)
                labels.append(loc_name)
    x_train = texts[:int(len(texts) * constant.training_ratio)]
    y_train = labels[:int(len(texts) * constant.training_ratio)]
    x_test = texts[int(len(texts) * constant.training_ratio):]
    y_test = labels[int(len(texts) * constant.training_ratio):]
    return x_train, y_train, x_test, y_test
def read_split_data_yelp(config):
    review_list = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/yelp/review_list", "rb"))
    loc_cat = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/yelp/loc_cat", "rb"))
    samples = list(zip(review_list, loc_cat))[:20000]
    random.shuffle(samples)
    constant = Constants_yelp()
    texts = [x[0] for x in samples]
    labels =[x[1] for x in samples]
    x_train = texts[:int(len(texts) * constant.training_ratio)]
    y_train = labels[:int(len(texts) * constant.training_ratio)]
    x_test = texts[int(len(texts) * constant.training_ratio):]
    y_test = labels[int(len(texts) * constant.training_ratio):]
    return x_train, y_train, x_test, y_test
def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def get_evaluation(y_true, y_prob, list_metrics, constant= Constants()):
    y_pred = np.argmax(y_prob, -1)
    print(len(y_true), y_true)
    print(len(y_pred), y_pred)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    if 'classfication_report' in list_metrics:
        output['classfication_report'] = metrics.classification_report(y_true, y_pred, target_names=constant.loc_list)
    return output


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return [int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]
def check_tensor(vector, name=None):
    try:
        if isinstance(vector, torch.Tensor):
            if not np.any(np.isnan(vector.cpu().detach().numpy())):
                return True
            print("[] is false".format(name))
            return False
        elif isinstance(vector, np.ndarray):
            if not np.any(np.isnan(vector)):
                return True
            print("[] is false".format(name))
            return False
        elif isinstance(vector, list):
            vector = np.asarray(vector, dtype=np.float32)
            if not np.any(np.isnan(vector)):
                return True
            print("[] is false".format(name))
            return False
    except Exception as ex:
        print(traceback.format_exc())
        print("name is [{}] value is {}".format(name, vector))
        return False

if __name__ == "__main__":
    # word, sent = get_max_lengths("../data/test.csv")
    # print (word)
    # print (sent)
    s = "The following are 30 code examples for showing how to use nltk.stem.snowball.SnowballStemmer(). These examples are extracted from open source projects."
    print(clean_str(s))





