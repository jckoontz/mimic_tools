import torch
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm,trange

class NER_MODEL(torch.nn.Module):

    def __init__(self, device, pretrained_path:str,
    output_dim:int, epochs:int, max_grad_norm, 
    FULL_FINETUNING:bool, model_name:str, n_gpu:int, batch_num):
        super(NER_MODEL, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(pretrained_path,num_labels=output_dim)
        self.device = device
        self.pretrained_path = pretrained_path
        self.model_name = model_name

        if FULL_FINETUNING:
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
            optimizer_grouped_parameters = [{"params": [p for _, p in param_optimizer]}]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.n_gpu = n_gpu
        if self.n_gpu > 1:
            self.bert = torch.nn.DataParallel(self.bert)
        self.batch_num = batch_num

    #foward pass
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.bert(input_ids, token_type_ids=None,
        attention_mask=attention_mask, labels=labels)
        return output

    def fit(self, train_dataloader):
        print('Training model')
        print(f'Batch size: {self.batch_num}')
        num_steps=len(train_dataloader) * self.epochs
        print(f'Number of steps: {num_steps}')
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
        num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.epochs)
        for _ in trange(self.epochs,desc="Epoch"):
            self.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                outputs = self.forward(b_input_ids, token_type_ids=None,
                attention_mask=b_input_mask, labels=b_labels)
                loss, _ = outputs[:2]
                if self.n_gpu>1:
                    # When multi gpu, average it
                    loss = loss.mean()
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=self.max_grad_norm)
                # update parameters
                self.optimizer.step()
                scheduler.step()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))