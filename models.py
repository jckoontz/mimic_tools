'''
NER model
'''
import torch
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW
from tqdm import trange


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

    def fit(self, train_dataloader):
        '''
        Train NER model
        '''
        epochs = self.config['epochs']
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(self.config['device'])

        model = self.train()
        losses = []
        for _ in trange(epochs, desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            train_loss = 0
            for _, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch = tuple(t.to(self.config['device']) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                logits = self.forward(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask)
                if b_input_mask is not None:
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.config['num_classes'])
                    active_labels = torch.where(
                        active_loss, b_labels.view(-1), torch.tensor(
                            loss_fn.ignore_index).type_as(b_labels)
                    )
                    loss = loss_fn(active_logits, active_labels)
                else:
                    loss = loss_fn(
                        logits.view(-1, self.config['num_classes'], b_labels.view(-1)))
                loss.backward()
                train_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            print("Train loss: {}".format(train_loss/nb_tr_steps))


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

    def fit(self, train_dataloader):

        epochs = self.config['epochs']
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(self.config['device'])

        model = self.train()
        losses = []
        for _ in trange(epochs, desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            train_loss = 0
            for _, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch = tuple(t.to(self.config['device']) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                logits = self.forward(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask)
                if b_input_mask is not None:
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.config['num_classes'])
                    active_labels = torch.where(
                        active_loss, b_labels.view(-1), torch.tensor(
                            loss_fn.ignore_index).type_as(b_labels)
                    )
                    loss = loss_fn(active_logits, active_labels)
                else:
                    loss = loss_fn(
                        logits.view(-1, self.config['num_classes'], b_labels.view(-1)))
                loss.backward()
                train_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
        print("Train loss: {}".format(train_loss/nb_tr_steps))
