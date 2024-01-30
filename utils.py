import pandas as pd
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

def read_data(filename='data.csv'):

    df = pd.read_csv(filename)
    records = df.to_dict('records')
    return records

def split_train_test(records, train_test_split=0.95):
    random.shuffle(records)
    train_ids = random.sample([n for n in range(len(records))], k=int(train_test_split*len(records)))
    train_records, test_records = [], []
    for i, row in enumerate(records):
        if i in train_ids:
            train_records.append(row)
        else:
            test_records.append(row)

    return train_records, test_records

def label_to_idx(records):
    label2idx = defaultdict(int)
    label2name = defaultdict(str)
    num_labels = 0
    for i, record in enumerate(records):
        if record['ONET'] not in label2name:
            label2name[record['ONET']] = record['ONET_NAME']
            label2idx[record['ONET']] = num_labels
            num_labels += 1
        records[i]['idx'] = label2idx[record['ONET']]
    return records, label2idx, label2name
def append_embeddings(records, title_embeddings, body_embeddings, title_body_embeddings):
    for i, row in enumerate(title_embeddings):
        records[i]['title_embedding'] = title_embeddings[i]
        records[i]['body_embedding'] = body_embeddings[i]
        records[i]['title_body_embedding'] = title_body_embeddings[i]
    return records

def get_labels(records, label2idx):
    labels = []

    for i, record in enumerate(records):
        labels.append(label2idx[record['ONET']])
    return labels

def flatten_preds(preds):
    flatten = []
    for pred in preds:
        flatten.append(pred[0])
    return flatten

def generate_scores(preds, labels):
    averaging = "weighted"
    print(f'prec: {precision_score(labels, preds, average=averaging)} '
          f'recall: {recall_score(labels, preds, average=averaging)} '
          f'f1: {f1_score(labels, preds, average=averaging)} '
          f'accuracy: {accuracy_score(labels, preds)}')



def get_metrics(preds, labels):
    res_title, res_body, res_title_body = preds

    res_title, res_body, res_title_body = flatten_preds(res_title), flatten_preds(res_body), flatten_preds(res_title_body)
    print('Generate scores only with title embedding....')
    generate_scores(res_title, labels)
    print('\nGenerate scores only with body embedding....')
    generate_scores(res_body, labels)
    print('\nGenerate scores with both title and body embedding.....')
    generate_scores(res_title_body, labels)