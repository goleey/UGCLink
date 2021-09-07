import csv
import sys

sys.path.append("../")
from config import alp_config
import json
import pickle
import time
from logging import getLogger
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from dataset import collate_fn, index_word
from utils_ssl import to_cuda
from tensorboardX import SummaryWriter
from torchsummary import summary
import sys

sys.path.insert(0, "../document_classification_word")
from hierarchical_att_model import HierAttNet

logger = getLogger()
writer = SummaryWriter()
max_number = torch.finfo(torch.float32).max
conf = alp_config()
class_num = 10


class MatchPyramidClassifier(object):

    def __init__(self, params):
        logger.info("Initializing GRUClassifier")
        self.params = params
        self.train_data = params.train_data
        self.test_data = params.test_data
        self.epoch_cnt = 0
        self.relevance_model = word_relevance_model(params)
        self.optimizer = torch.optim.Adam(
            list(self.relevance_model.parameters()),
            lr=self.params.lr,
            weight_decay = 0.001
        )
        self.vocab_size = len(params.word2idx)
        self.relevance_model.cuda()
        # for name, value in self.matchPyramid.named_parameters():
        #     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
        # exit()

    def run(self):
        max_acc = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_auc = 0
        for i in range(self.params.n_epochs):
            self.train()
            acc, precision, recall, f1, auc_score = self.evaluate()
            if max_acc < acc:
                max_acc = acc
                max_precision = precision
                max_recall = recall
                max_f1 = f1
                max_auc = auc
            self.epoch_cnt += 1
        print(max_acc, max_precision, max_recall, max_f1, max_auc)

    def train(self):
        logger.info("Training in epoch %i" % self.epoch_cnt)
        self.relevance_model.train()
        data_loader = DataLoader(self.train_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 )
        pred_list = list()
        label_list = list()
        loss_list = list()
        for data_iter in data_loader:
            pair, f_user_text, t_user_text, text_len, labels = data_iter
            f_word_idx, t_word_idx, text_len, labels = \
                index_word(pair, f_user_text, t_user_text, text_len, labels, word2idx=self.params.word2idx,
                           max_seq_length=self.params.max_seq_len,
                           max_word_length_f=self.params.max_word_len_f, max_word_length_t=self.params.max_word_len_t)
            f_word_idx, t_word_idx, text_len, labels = to_cuda(f_word_idx,
                                                               t_word_idx,
                                                               text_len,
                                                               labels)
            foursquare_loc_dis, twitter_loc_dis, doc1_embedding, doc2_embedding, mp_output = self.relevance_model(f_word_idx, t_word_idx)
            loss = F.cross_entropy(mp_output, labels) + 0.5* F.mse_loss(foursquare_loc_dis, doc1_embedding) + 0.5*F.mse_loss(twitter_loc_dis, doc2_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predictions = mp_output.data.max(1)[1]
            pred_list.extend(predictions.tolist())
            label_list.extend(labels.tolist())
            loss_list.append(loss.detach().cpu().numpy())
            print(mp_output.data)
            print(predictions)
            print(labels.data.tolist())
            # exit()

        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        losses = np.mean(loss_list)

        writer.add_scalar("train acc", acc, self.epoch_cnt)
        writer.add_scalar("train f1", f1, self.epoch_cnt)
        writer.add_scalar("train losses", losses, self.epoch_cnt)
        logger.info("Train loss in epoch %i :%.4f" % (self.epoch_cnt, losses))
        logger.info("Train ACC score in epoch %i :%.4f" % (self.epoch_cnt, acc))
        logger.info("Train F1 score in epoch %i :%.4f" % (self.epoch_cnt, f1))

    def evaluate(self):
        logger.info("Evaluating in epoch %i" % self.epoch_cnt)
        self.relevance_model.cuda()
        data_loader = DataLoader(self.test_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True)
        pred_list = list()
        label_list = list()
        loss_list = list()
        with torch.no_grad():
            for i, data_iter in enumerate(data_loader):
                pair, f_user_text, t_user_text, text_len, labels = data_iter
                f_word_idx, t_word_idx, text_len, labels = \
                    index_word(pair, f_user_text, t_user_text, text_len, labels, word2idx=self.params.word2idx,
                               max_seq_length=self.params.max_seq_len,
                               max_word_length_f=self.params.max_word_len_f, max_word_length_t=self.params.max_word_len_t)
                f_word_idx, t_word_idx, text_len, labels = to_cuda(f_word_idx,
                                                                   t_word_idx,
                                                                   text_len,
                                                                   labels)
                foursquare_loc_dis, twitter_loc_dis, doc1_embedding, doc2_embedding, mp_output = self.relevance_model(
                    f_word_idx, t_word_idx)
                loss = F.cross_entropy(mp_output, labels) + 0.5* F.mse_loss(foursquare_loc_dis, doc1_embedding) + 0.5* F.mse_loss(
                    twitter_loc_dis, doc2_embedding)

                predictions = mp_output.data.max(1)[1]
                pred_list.extend(predictions.tolist())
                label_list.extend(labels.tolist())
                loss_list.append(loss.cpu().numpy())
        acc = accuracy_score(label_list, pred_list)
        precision = precision_score(label_list, pred_list)
        recall = recall_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        fpr, tpr, thresholds = roc_curve(label_list, pred_list, pos_label=1)
        auc_score = auc(fpr, tpr)
        losses = np.mean(loss_list)
        writer.add_scalar("test acc", acc, self.epoch_cnt)
        writer.add_scalar("test f1", f1, self.epoch_cnt)
        writer.add_scalar("test losses", losses, self.epoch_cnt)
        logger.info("Test loss in epoch %i :%.4f" % (self.epoch_cnt, losses))
        logger.info("Test ACC score in epoch %i :%.4f" % (self.epoch_cnt, acc))
        logger.info("Test F1 score in epoch %i :%.4f" % (self.epoch_cnt, f1))
        return (acc, precision, recall, f1, auc_score)

class gru_model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_size = params.mp_dim_out
        self.hidden_size = params.gru_hidden
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        self.fc = torch.nn.Linear(2 * self.hidden_size, params.dim_out)
        # self.fc = torch.nn.Linear(2 * self.params.max_seq_len * self.hidden_size, params.dim_out)

    def forward(self, input):
        input = torch.transpose(input, 0, 1)
        hidden_state = self._init_hidden_state(input.size()[1], class_num)
        f_output, h_output = self.gru(input, hidden_state)
        # f_output = torch.transpose(f_output, 0, 1)
        # f_output = f_output.contiguous().view(self.params.batch_size*self.params.max_seq_len, 2 * self.hidden_size)
        h_output = torch.transpose(h_output, 0, 1)
        h_output = h_output.contiguous().view(-1, 2 * self.hidden_size)
        output = self.fc(h_output)
        # print(output.size())
        return output

    def _init_hidden_state(self, batch_size, hidden_size):
        word_hidden_state = torch.zeros(2, batch_size, hidden_size)
        if torch.cuda.is_available():
            word_hidden_state = word_hidden_state.cuda()
        return word_hidden_state


class MatchPyramid(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.max_len_f = params.max_word_len_f
        self.max_len_t = params.max_word_len_t
        self.conv1_size = [int(_) for _ in params.conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in params.pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in params.conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in params.pool2_size.split("_")]
        self.dim_hidden = params.mp_hidden
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1],
                                       self.dim_hidden, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(self.dim_hidden, params.mp_dim_out, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear2.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.dim_hidden}))

    def forward(self, x1, x2):
        # x1,x2:[batch, seq_len, dim_xlm]

        bs, seq_len_f, dim_xlm = x1.size()
        _, seq_len_t, _ = x2.size()
        pad1 = self.max_len_f - seq_len_f
        pad2 = self.max_len_t - seq_len_t
        # simi_img:[batch, 1, seq_len, seq_len]
        # cosine similarity
        x1_norm = x1.norm(dim=-1, keepdim=True)
        x1_norm = x1_norm + 1e-8
        x2_norm = x2.norm(dim=-1, keepdim=True)
        x2_norm = x2_norm + 1e-8
        x1 = x1 / x1_norm
        x2 = x2 / x2_norm
        simi_img = torch.matmul(x1, x2.transpose(1, 2))
        # print(simi_img.size())
        # exit()
        # print(simi_img)
        # print("====")
        # print(simi_img.detach().cpu().numpy().tolist())
        # exit()
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len_f, self.max_len_t)
        simi_img = torch.where(simi_img>0.3, simi_img, torch.zeros_like(simi_img).cuda())
        # print(simi_img.detach().cpu().numpy())
        print(np.any(simi_img.detach().cpu().numpy()!=0.0))
        # exit()
        # print("conv1", self.conv1.weight)
        # print("conv2", self.conv2.weight)
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]

        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        # print("conv1", simi_img.detach().cpu().numpy().tolist())
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        simi_img = self.pool2(simi_img)
        # print("conv2", simi_img.detach().cpu().numpy().tolist())
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # output = self.linear1(simi_img)
        output = F.relu(self.linear1(simi_img))
        # print(output.detach().cpu().numpy().tolist())
        output = self.linear2(output)
        # print(output.detach().cpu().numpy().tolist())
        print(output.size())
        return output

class MatchPyramid_identity(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.max_len_f = params.max_word_len_f
        self.max_len_t = params.max_word_len_t
        self.conv1_size = [int(_) for _ in params.conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in params.pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in params.conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in params.pool2_size.split("_")]
        self.dim_hidden = params.mp_hidden
        self.mp_dim_out = params.mp_dim_out
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        # torch.nn.init.ones_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.ones_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1],
                                       self.mp_dim_out, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.dim_hidden}))
    def forward(self, x1, x2):
        # x1,x2:[batch, seq_len, dim_xlm]

        bs, seq_len_f, dim_xlm = x1.size()
        _, seq_len_t, _ = x2.size()
        pad1 = self.max_len_f - seq_len_f
        pad2 = self.max_len_t - seq_len_t
        # simi_img:[batch, 1, seq_len, seq_len]
        # cosine similarity
        x1_norm = x1.norm(dim=-1, keepdim=True)
        x1_norm = x1_norm + 1e-8
        x2_norm = x2.norm(dim=-1, keepdim=True)
        x2_norm = x2_norm + 1e-8
        x1 = x1 / x1_norm
        x2 = x2 / x2_norm
        simi_img = torch.matmul(x1, x2.transpose(1, 2))
        # print(simi_img.size())
        # exit()
        # print(simi_img)
        # print("====")
        # print(simi_img.detach().cpu().numpy().tolist())
        # exit()
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len_f, self.max_len_t)
        print(simi_img.detach().cpu().numpy().tolist())
        simi_img = torch.where(simi_img>0.9, simi_img, torch.zeros_like(simi_img).cuda())
        print(simi_img.detach().cpu().numpy().tolist())
        #exist or not?
        print(simi_img.size())
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        print(simi_img.size())
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        print(simi_img.size())
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        # print("conv1", simi_img.detach().cpu().numpy().tolist())
        print(simi_img.size())
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        print(simi_img.size())
        simi_img = self.pool2(simi_img)
        print(simi_img.size())
        # print("conv2", simi_img.detach().cpu().numpy().tolist())
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        print(simi_img.size())
        simi_img = simi_img.squeeze(1).view(bs, -1)
        print(simi_img.size())
        output = self.linear1(simi_img)
        # print(output.detach().cpu().numpy().tolist())
        print(output.size())
        # exit()
        return output

class word_relevance_model(torch.nn.Module):
    def __init__(self, params):
        super(word_relevance_model, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=params.embedding_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict = torch.from_numpy(dict.astype(np.float32))
        self.lookup = torch.nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,freeze=False)
        # self.matchparamid = MatchPyramid(params)
        self.doc_fc = torch.nn.Linear(embed_size, class_num)
        self.kd_model = params.kd_model
        # for name, parameters in self.kd_model.named_parameters():
        #     print(name, parameters.requires_grad)
        # exit()
        self.matchparamid = MatchPyramid_identity(params)
        self.gru = gru_model(params)
        self.params = params
        # self.word_rev = torch.nn.ModuleList([self.gru, self.matchparamid])

    def forward(self, x1, x2):

        sen1_embedding, sen2_embedding = self.lookup(x1), self.lookup(x2)
        doc1_embedding, doc2_embedding = torch.mean(sen1_embedding,dim=-2), torch.mean(sen2_embedding, dim=-2)

        batch_size, seq_len, word_len_f, emb_size = sen1_embedding.size()
        _, _, word_len_t, _ = sen2_embedding.size()

        sen1_embedding = sen1_embedding.view(batch_size * seq_len, word_len_f, emb_size)
        sen2_embedding = sen2_embedding.view(batch_size * seq_len, word_len_t, emb_size)
        doc1_embedding = doc1_embedding.view(batch_size*seq_len, emb_size)
        doc2_embedding = doc2_embedding.view(batch_size*seq_len, emb_size)
        doc1_embedding = self.doc_fc(doc1_embedding)
        doc2_embedding = self.doc_fc(doc2_embedding)
        # doc1_embedding = doc1_embedding.view(batch_size , seq_len, class_num)
        # doc2_embedding = doc2_embedding.view(batch_size , seq_len, class_num)
        sen1_idx = x1.view(batch_size * seq_len, word_len_f)
        sen2_idx = x2.view(batch_size * seq_len, word_len_t)
        foursquare_loc_dis, _ = self.kd_model("", sen1_idx)
        twitter_loc_dis, _ = self.kd_model("", sen2_idx)
        # foursquare_loc_dis = foursquare_loc_dis.view(batch_size, seq_len, class_num)
        # twitter_loc_dis = twitter_loc_dis.view(batch_size, seq_len, class_num)
        # loc_dis = jsd(foursquare_loc_dis, twitter_loc_dis)
        matchparamid_output = self.matchparamid(sen1_embedding, sen2_embedding)
        matchparamid_output = matchparamid_output.view(batch_size, seq_len, self.params.mp_dim_out)
        mp_output = self.gru(matchparamid_output)
        # print(foursquare_loc_dis.size(), twitter_loc_dis.size(), doc1_embedding.size(), doc2_embedding.size())
        # exit()
        return foursquare_loc_dis, twitter_loc_dis, doc1_embedding, doc2_embedding, mp_output

# tensor1: [bs,distribution]
def jsd(tensor1, tensor2):
    tensor1 = F.softmax(tensor1, dim=1)
    tensor2 = F.softmax(tensor2, dim=1)
    m_tensor = (tensor1 + tensor2) * 0.5
    kl_1 = F.kl_div(torch.log(m_tensor), tensor1, reduction="none")
    kl_2 = F.kl_div(torch.log(m_tensor), tensor2, reduction="none")
    kl_1 = torch.sum(kl_1, dim=1)
    kl_2 = torch.sum(kl_2, dim=1)
    return (kl_1 + kl_2) * 0.5

def consine_sim(tensor1, tensor2):
    return F.cosine_similarity(tensor1, tensor2, dim=-1)