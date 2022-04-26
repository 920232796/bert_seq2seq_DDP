
import torch
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Predictor
import numpy as np

def compute_similarity(in_1, in_2):
    res = np.dot(in_1, in_2) / (np.linalg.norm(in_1) * np.linalg.norm(in_2))
    return res

maxlen = 256
model_name = "bert" # 选择模型名字
task_name = "embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "../state_dict/bert-base-chinese/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/bert-base-chinese/pytorch_model.bin" # roberta模型位置

tokenizer = Tokenizer(vocab_path, do_lower_case=True, max_len=maxlen)
bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
bert_model.load_pretrain_params(model_path, strict=False)

predictor = Predictor(bert_model, tokenizer)
text = ["今天天气很好", "今天天气不错", "今天有事出去忙"]

embedding_1 = predictor.predict_embedding(text[0], maxlen=maxlen)
embedding_2 = predictor.predict_embedding(text[1], maxlen=maxlen)
embedding_3 = predictor.predict_embedding(text[2], maxlen=maxlen)

print(f"cos sim 1-2 is {compute_similarity(embedding_1, embedding_2)}")
print(f"cos sim 1-3 is {compute_similarity(embedding_1, embedding_3)}")





