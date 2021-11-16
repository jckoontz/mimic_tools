'''
NER models
'''
import torch.nn as nn
from transformers import BertModel


class NER(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config['pretrained_model'])
        self.dropout = nn.Dropout(p=self.config['dropout'])
        self.out = nn.Linear(self.bert.config.hidden_size,
                             self.config['num_classes'])

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        output = self.dropout(output[0])
        return self.out(output)


class NER_LSTM(nn.Module):
    def __init__(self, config: dict):
        super(NER_LSTM, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config['pretrained_model'])
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=self.config['dropout'])
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            bidirectional=True, batch_first=True)
        self.out = nn.Linear(self.hidden_size*2, self.config['num_classes'])

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        lstm_output, (h, c) = self.lstm(output['last_hidden_state'])
        return self.out(self.dropout(lstm_output))
