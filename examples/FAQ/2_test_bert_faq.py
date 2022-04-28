import torch
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Predictor
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
import faiss

faq_data_path = "../data/financezhidao_filter.csv"
answer_save_path = "../data/finance_fqa.json"
embeddings_save_path = "../data/finance_embeddings.json"

maxlen = 256
d = 768
nlist = 5

model_name = "bert" # 选择模型名字
task_name = "embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta/pytorch_model.bin" # roberta模型位置

tokenizer = Tokenizer(vocab_path, do_lower_case=True, max_len=maxlen)
bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
bert_model.load_pretrain_params(model_path)

predictor = Predictor(bert_model, tokenizer)

class Search:
    def __init__(self, training_vectors, d, nlist=10, nprobe=1):
        quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not self.index.is_trained
        self.index.train(training_vectors)
        assert self.index.is_trained
        self.index.nprobe = nprobe  # default nprobe is 1, try a few more
        self.index.add(training_vectors)  # add may be a bit slower as well
        self.d = d

    def search(self, answer, query, k=10):
        query = query.numpy().reshape(-1, self.d)
        D, I = self.index.search(query, k)  # actual search
        result = []
        all_question = list(answer.keys())
        for s, i in zip(D[0], I[0]):
            print(i)
            if i != -1:
                result.append({all_question[i]: s})

        print(result)

if __name__ == '__main__':
    # load data
    answer = torch.load(answer_save_path)
    embeddings = torch.load(embeddings_save_path)

    method = Search(training_vectors=embeddings, d=d, nlist=nlist, nprobe=2)

    while True:
        question = input("请输入问题：")
        if question == "q":
            break
        question_embedding = predictor.predict_embedding(question, maxlen=maxlen)
        method.search(answer, question_embedding, k=10)





