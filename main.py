# 从word_relevance_model修改而来，过滤推特的文本，只保留tf-idf有用的部分
import os
import sys
import torch
import numpy as np
import argparse
sys.path.append("../")
from document_classification_word.hierarchical_att_model import HierAttNet
from utils_ssl import init_logger, load_w2v
import model
import model_mlp
from dataset import dataset

from word_relevance_filtered_model import model_identity, model_kd, model_only_kd, model_kd_embedding, model_kd_embedding_delta_t, model_kd_embedding_thre_r, model_kd_embedding_coe_lamda, model_kd_embedding_wo_tm, model_kd_embedding_training_ratio
from word_relevance_filtered_model import model_identity_all_units, model_kd_embedding_casestudy
from config import alp_config
import os
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(
    description='Train and Evaluate MatchPyramid on MSRP dataset')

# main parameters
parser.add_argument("--training_ratio", type=float, default=1.0,
                    help="")
parser.add_argument("--delta_t", type=int, default=5,
                    help="")
parser.add_argument("--filter_threshold", type=float, default=0.9,
                    help="")
parser.add_argument("--lamda", type=float, default=0.5,
                    help="")
parser.add_argument("--data_path", type=str, default="/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/",
                    help="")
parser.add_argument("--dump_path", type=str, default="./dump/",
                    help="")
parser.add_argument("--embedding_path", type=str,
                    default="/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/glove.6B.50d.txt",
                    help="")
# parser.add_argument("--filter_word_method", type=str, choices=["tfidf", "lda", "dc"], default="dc",
#                     help="")
parser.add_argument("--max_seq_len", type=int, default=10,
                    help="")
parser.add_argument("--max_word_len_f", type=int, default=10,
                    help="")
parser.add_argument("--max_word_len_t", type=int, default=300,
                    help="")
# parser.add_argument("--max_word_len", type=int, default=10,
#                     help="")
parser.add_argument("--batch_size", type=int, default=32,
                    help="")
parser.add_argument("--lr", type=float, default=0.01,
                    help="")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="")
# convolution model parameters
# parser.add_argument("--conv1_size", type=str, default="5_5_8",
#                     help="")
# parser.add_argument("--pool1_size", type=str, default="5_5",
#                     help="")
# parser.add_argument("--conv2_size", type=str, default="3_3_16",
#                     help="")
# parser.add_argument("--pool2_size", type=str, default="5_5",
#                     help="")
parser.add_argument("--conv1_size", type=str, default="3_3_1",
                    help="")
parser.add_argument("--pool1_size", type=str, default="20_20",
                    help="")
parser.add_argument("--conv2_size", type=str, default="3_3_8",
                    help="")
parser.add_argument("--pool2_size", type=str, default="2_2",
                    help="")
parser.add_argument("--mp_hidden", type=int, default=128,
                    help="")
# parser.add_argument("--mp_dim_out", type=int, default="10",
#                     help="")
parser.add_argument("--mp_dim_out", type=int, default="1",
                    help="")
# gru model parameters
parser.add_argument("--dim_embedding", type=int, default=50,
                    help="")
parser.add_argument("--gru_hidden", type=int, default=10,
                    help="")
parser.add_argument("--dim_mapping_out", type=int, default=1,
                    help="")
#final mp

parser.add_argument("--dim_out", type=int, default=2,
                    help="")
parser.add_argument("--model", type=str, choices=["model", "model_mlp", "model_identity", "model_identity_all_units", "model_only_kd", "model_kd", "model_kd_embedding", "model_kd_embedding_delta_t", "model_kd_embedding_thre_r", "model_kd_embedding_coe_lamda","model_kd_embedding_wo_tm", "model_kd_embedding_training_ratio", "model_kd_embedding_casestudy"], default="model_kd_embedding",
                    help="")

parser.add_argument("--kd_model", type=str, choices=["foursquare_dc_model"], default="foursquare_dc_model",
                    help="")
# parse arguments
params = parser.parse_args()

# check parameters

logger = init_logger(params)

params.word2idx, params.glove_weight = load_w2v(params.embedding_path, params.dim_embedding)

conf = alp_config(params.data_path)
train_data = dataset(conf.x_train, conf.y_train, conf.foursquare_checkins, conf.twitter_uid_tweeets, params.delta_t)
test_data = dataset(conf.x_test, conf.y_test, conf.foursquare_checkins, conf.twitter_uid_tweeets, params.delta_t)

params.train_data = train_data
params.test_data = test_data
if params.kd_model == "foursquare_dc_model":
    params.kd_model = HierAttNet(50, 50, 128, 10,
                       "/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/glove.6B.50d.txt", 5, 300)
    params.kd_model.load_state_dict(torch.load("/home/shenhuawei/gaohao/mulmodal-alp-mlp/document_classification_word/foursquare/han_model")["state_dict"])
    for name, parameters in params.kd_model.named_parameters():
        # print(name, parameters.requires_grad)
        parameters.requires_grad = False

if params.model == "model":
    mp_model = model.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_mlp":
    mp_model = model_mlp.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_identity":
    mp_model = model_identity.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_identity_all_units":
    mp_model = model_identity_all_units.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd":
    mp_model = model_kd.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_only_kd":
    mp_model = model_only_kd.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding":
    mp_model = model_kd_embedding.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_delta_t":
    print(params.delta_t)
    mp_model = model_kd_embedding_delta_t.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_thre_r":
    mp_model = model_kd_embedding_thre_r.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_coe_lamda":
    mp_model = model_kd_embedding_coe_lamda.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_wo_tm":
    mp_model = model_kd_embedding_wo_tm.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_training_ratio":
    mp_model = model_kd_embedding_training_ratio.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_kd_embedding_casestudy":
    mp_model = model_kd_embedding_casestudy.MatchPyramidClassifier(params)
    mp_model.run()
