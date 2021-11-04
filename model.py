import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from seqeval.metrics import classification_report, accuracy_score, f1_score


class NER_MODEL(torch.nn.Module):

    def __init__(self, device, pretrained_path: str,
                 output_dim: int, epochs: int, max_grad_norm,
                 full_finetune: bool, model_name: str, n_gpu: int, batch_num: int):
        super(NER_MODEL, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            pretrained_path, num_labels=output_dim)
        self.device = device
        self.pretrained_path = pretrained_path
        self.model_name = model_name

        if full_finetune:
            # Fine tune model all layer parameters
            param_optimizer = list(self.bert.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
        else:
            # Only fine tune classifier parameters
            param_optimizer = list(self.bert.classifier.named_parameters())
            optimizer_grouped_parameters = [
                {"params": [p for _, p in param_optimizer]}]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.n_gpu = n_gpu
        if self.n_gpu > 1:
            self.bert = torch.nn.DataParallel(self.bert)
        self.batch_num = batch_num

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.bert(input_ids, token_type_ids=None,
                           attention_mask=attention_mask, labels=labels)
        return output

    def fit(self, train_dataloader):
        print('Training model')
        print(f'Batch size: {self.batch_num}')
        num_steps = len(train_dataloader) * self.epochs
        print(f'Number of steps: {num_steps}')
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.epochs)
        for _ in trange(self.epochs, desc="Epoch"):
            self.train()
            train_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for _, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                outputs = self.forward(b_input_ids, token_type_ids=None,
                                       attention_mask=b_input_mask, labels=b_labels)
                loss, _ = outputs[:2]
                if self.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()

                train_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.parameters(), max_norm=self.max_grad_norm)
                # update parameters
                self.optimizer.step()
                scheduler.step()

        print("Train loss: {}".format(train_loss/nb_tr_steps))

    def evaluate(self, valid_dataloader):

        y_true = []
        y_pred = []
        labels = set()
        filter_list = ['X', '[CLS]', '[SEP]']
        self.eval()

        print("Starting evaluation")
        for _, batch in enumerate(valid_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.forward(b_input_ids, token_type_ids=None,
                                       attention_mask=b_input_mask, labels=b_labels)
                logits = outputs[1]
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()

            label_ids = b_labels.to('cpu').numpy()
            # Only predict the real word, mark=0, will not calculate
            input_mask = b_input_mask.to('cpu').numpy()

            # Compare the valuable predict result
            for i, mask in enumerate(input_mask):
                # Real one
                real = []
                # Predict one
                predicted = []
                for j, m in enumerate(mask):
                    if m:
                        # Exclude the X label
                        if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" \
                                and tag2name[label_ids[i][j]] != "[SEP]":
                            real.append(tag2name[label_ids[i][j]])
                            if tag2name[logits[i][j]] in filter_list:
                                predicted.append('O')
                                labels.add('O')
                            else:
                                predicted.append(tag2name[logits[i][j]])
                                labels.add(tag2name[logits[i][j]])
                    else:
                        break

                y_true.append(real)
                y_pred.append(predicted)

        report = classification_report(y_true, y_pred, digits=4)
        print("***** Eval results *****")
        print("\n%s" % (report))
        print("f1 socre: %f" % (f1_score(y_true, y_pred)))
        print("Accuracy score: %f" % (accuracy_score(y_true, y_pred)))
