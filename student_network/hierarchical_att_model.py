"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import sys
# sys.path.append("../")
from document_classification_word.word_att_model import WordAttNet

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size, num_classes)

    def _init_hidden_state(self, batch_size):
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()

    def forward(self, raw, input):
        self._init_hidden_state(input.size()[0])
        output, word_hidden_state, alpha = self.word_att_net(raw, input.permute(1, 0), self.word_hidden_state)
        # print("parameters")
        # print(self.word_att_net.gru.weight_ih_l0.cpu().detach().tolist())
        # print(self.word_att_net.gru.weight_hh_l0.cpu().detach().tolist())
        # print(self.word_att_net.gru.bias_ih_l0.cpu().detach().tolist())
        # print(self.word_att_net.gru.bias_hh_l0.cpu().detach().tolist())
        return output, alpha
