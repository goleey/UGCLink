import os
import random
import json
import time
import sys
sys.path.append("../")
from config import alp_config
import pickle as pkl
import os
import sys
import logging
import time
from copy import deepcopy
from datetime import timedelta, datetime
import torch
from torchtext.vocab import GloVe, Vectors

conf = alp_config()
# def filter_data(info_in_same_day):
#     new_dic = {}
#     for k, v in info_in_same_day.items():
#         if len(v) != 0:
#             new_dic[k] = v
#     return new_dic

def negative_sample():
    info_in_same_day = pkl.load(open(conf.info_in_same_day, "rb"),encoding="bytes")
    neg_num = conf.train_neg_ratio
    def _sample_negative_samples(info_in_same_day, neg_num):
        neg_dic = {}
        for k, v in info_in_same_day.items():
            print(k)
            if len(v) != 0:
                neg_list = []
                time1 = time.time()
                while len(neg_list) < neg_num:
                    sample = random.sample(info_in_same_day.items(), 1)[0]
                    key = sample[0]
                    value = sample[1]
                    if key == k:
                        continue
                    else:
                        days_1 = set(v.keys())
                        days_2 = set(value.keys())
                        if len(days_1&days_2) >= 1:
                            neg_list.append(key)
                    time2 = time.time()
                    if time2 - time1 >= 100:
                        break
                if len(neg_list) != 0:
                    neg_dic[k] = neg_list
        return neg_dic

    neg_dic = _sample_negative_samples(info_in_same_day, neg_num)
    if not os.path.exists(conf.neg_dict):
        pkl.dump(neg_dic, open(conf.neg_dict, "wb+"))
    # print(neg_dic)
    return neg_dic
def split_data(train_ratio):
    neg_dic = pkl.load(open(conf.neg_dict, "rb"))
    x_pos = []
    neg_list_dic = {}
    # construct neg samples from foursquare to twitter(retrieve twitter false users)
    for k,v in neg_dic.items():
        x_pos.append(k)
        for neg in v:
            neg_list_dic.setdefault(k, []).append((k[0], neg[1]))
    random.shuffle(x_pos)
    train_pair = x_pos[:int(len(x_pos)*train_ratio)]
    test_pair = x_pos[int(len(x_pos)*train_ratio):]

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for pair in train_pair:
        x_train.append(pair)
        x_train.extend(neg_list_dic[pair])
        y_train.append(1)
        y_train.extend([0] * len(neg_list_dic[pair]))
    for pair in test_pair:
        x_test.append(pair)
        x_test.extend(neg_list_dic[pair])
        y_test.append(1)
        y_test.extend([0] * len(neg_list_dic[pair]))
    # for i in range(len(x_train)):
    #     print(x_train[i])
    #     print(y_train[i])
    pkl.dump(x_train, open(conf.x_train, "wb+"))
    pkl.dump(y_train, open(conf.y_train, "wb+"))
    pkl.dump(x_test, open(conf.x_test, "wb+"))
    pkl.dump(y_test, open(conf.y_test, "wb+"))
class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def init_logger(params):
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    command = ' '.join(command)
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("Running command: %s" % command)
    logger.info("")
    logger_scores = create_logger(os.path.join(params.dump_path, 'scores.log'), rank=getattr(params, 'global_rank', 0))

    return logger, logger_scores


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def load_w2v(data_path, dim_embedding):
    # glove = GloVe(dim=dim_embedding)
    glove = Vectors(data_path, ".")
    vocab_size = len(glove.stoi)
    word2idx = glove.stoi
    weight = deepcopy(glove.vectors)
    weight = torch.cat((weight, weight.new(1, dim_embedding).fill_(0.0)), dim=0)
    return word2idx, weight
def datetime2stamp(date_time):
    datetime_stamp = datetime.timestamp(date_time)
    return datetime_stamp


if __name__ == '__main__':
    # negative_sample()
    split_data(conf.train_ratio)