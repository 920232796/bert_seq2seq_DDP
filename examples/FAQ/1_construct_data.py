## 构建数据库
## 数据来源 https://github.com/murufeng/ChineseNlpCorpus
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
model_name = "bert" # 选择模型名字
task_name = "embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta/pytorch_model.bin" # roberta模型位置

tokenizer = Tokenizer(vocab_path, do_lower_case=True, max_len=maxlen)
bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
bert_model.load_pretrain_params(model_path)

predictor = Predictor(bert_model, tokenizer)

def resave_data():
    answer = collections.OrderedDict()
    embeddings = []
    df = pd.read_csv(faq_data_path)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if type(row[0]) == str:
            if row[0] not in answer:
                answer[row[0]] = row[2]
                embeddings.append(predictor.predict_embedding(row[0], maxlen=maxlen).numpy())

    embeddings = np.array(embeddings)
    torch.save(answer, answer_save_path)
    torch.save(embeddings, embeddings_save_path)

    print(f"数据保存成功: {answer_save_path}, {embeddings_save_path}")

if __name__ == '__main__':

    resave_data()