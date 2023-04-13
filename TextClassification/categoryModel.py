#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:07:47 2023

@author: hyeontaemin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:59:00 2023

@author: hyeontaemin
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from flask import jsonify

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
log_interval = 100
learning_rate =  5e-5


device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=12,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


model = BERTClassifier(bertmodel, dr_rate = 0.5).to(device)

#model = torch.load("sample_databertmodel5.pt",map_location=device)

model.load_state_dict(torch.load('TextClassification/sample_datamodelstate.pt',map_location=device))


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("채소")
            elif np.argmax(logits) == 1:
                test_eval.append("과일")
            elif np.argmax(logits) == 2:
                test_eval.append("정육")
            elif np.argmax(logits) == 3:
                test_eval.append("빵/떡")
            elif np.argmax(logits) == 4:
                test_eval.append("과자/음료/간식")
            elif np.argmax(logits) == 5:
                test_eval.append("앙념/오일/조미료/가루/견과류")
            elif np.argmax(logits) == 6:
                test_eval.append("유제품/계란")
            elif np.argmax(logits) == 7:
                test_eval.append("수산/건어물")
            elif np.argmax(logits) == 8:
                test_eval.append("면/통조림/가공식품")
            elif np.argmax(logits) == 9:
                test_eval.append("커피/원두/차")
            elif np.argmax(logits) == 10:
                test_eval.append("반찬")
            elif np.argmax(logits) == 11:
                test_eval.append("담배/주류")
            

        categoryPred = {"category" : test_eval[0]}
        return jsonify(categoryPred)
        

