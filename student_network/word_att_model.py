"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from document_classification_word.utils import matrix_mul, element_wise_mul, check_tensor
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50, num_classes=10):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict = torch.from_numpy(dict.astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # print(self.word_weight)
        # print(self.word_bias)
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.word_bias.data.normal_(mean,std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, raw, input, hidden_state):
        # print(hidden_state.size())
        # print("checking input")
        # if not check_tensor(input):
        #     print(raw)
        #     exit()
        output = self.lookup(input)
        # print(output.size())
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # print(f_output.size())
        # print("checking f_output")
        # if not check_tensor(f_output):
        #     print(raw)
        #     exit()
        # print("checking h_output")
        # if not check_tensor(h_output):
        #     print(raw)
        #     exit()
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        # print(output.size())
        output = matrix_mul(output, self.context_weight).permute(1,0)
        # print(output.size())
        output = F.softmax(output, dim=-1)
        # print("checking softmax")
        # if not check_tensor(output):
        #     print(raw)
        #     pprint(input.cpu().detach().numpy().tolist())
        #     pprint("---")
        #     pprint(output.cpu().detach().numpy().tolist())
        #     exit()
        alpha = output
        # print(output.size())
        output = element_wise_mul(f_output,output.permute(1,0))
        # print(output.size())
        output = self.fc(output)
        # print(output.size())
        output = torch.squeeze(output,dim=0)
        # print(output.size())
        # exit()
        return output, h_output, alpha


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
