# https://tianchi.aliyun.com/competition/entrance/531851/information
import torch
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq.dataset import bert_cls_collate_fn
from bert_seq2seq.trainer import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from bert_seq2seq import Predictor
import os

target = [0, 1]
train_path = "../data/语义匹配/train.tsv"
model_name = "roberta" # 选择模型名字
task_name = "classification"
vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta/pytorch_model.bin" # roberta模型位置
model_save_path = "./bert_semantic_matching.bin"
batch_size = 16
lr = 1e-5
# 加载字典
tokenizer = Tokenizer(vocab_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(epoches=3,
                  val_every_step=100,
                  batch_size=16,
                  env_type="pytorch",
                  device=device)

bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name, target_size=len(target))
## 加载预训练的模型参数～
bert_model.load_pretrain_params(model_path)
# 声明需要优化的参数
predictor = Predictor(bert_model, tokenizer)

def read_corpus(data_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        if len(line) == 3:
            sents_tgt.append(int(line[2]))
            sents_src.append(line[0] + "#" +line[1])

    return sents_src, sents_tgt

## 加载数据
all_input, all_label = read_corpus(train_path)
train_input, val_input, train_label, val_label = train_test_split(all_input, all_label, train_size=0.8, random_state=123)


## 自定义dataset
class SemanticMatchingDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(SemanticMatchingDataset, self).__init__()
        # 读原始数据
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        tokenizer_out = tokenizer.encode_plus(src)

        output = {
            "input_ids": tokenizer_out["input_ids"],
            "token_type_ids": tokenizer_out["token_type_ids"],
            "labels": tgt
        }
        return output

    def __len__(self):
        return len(self.sents_src)

class Evaluator:
    def __init__(self):
        self.best_acc = 0.0

    def on_validation(self, data):
        loss = data["loss"]
        step = data["iteration"]
        res = []
        for data in val_input:
            pred = predictor.predict_cls_classifier(data)
            pred = pred.argmax(dim=0).numpy()
            res.append(pred)

        f1 = f1_score(val_label, res)
        accuracy = accuracy_score(val_label, res)
        recall = recall_score(val_label, res)
        precision = precision_score(val_label, res)

        print(f" f1 is {f1}, acc is {accuracy}, recall is {recall} precision is {precision}")

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save(bert_model.state_dict(), model_save_path)
            print(f"模型保存成功～")


def main():

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=lr, weight_decay=1e-3)
    train_dataset = SemanticMatchingDataset(train_input, train_label)

    trainer.train(bert_model, optimizer=optimizer,
                  train_dataset=train_dataset,
                  evaluator=Evaluator,
                  collate_fn=bert_cls_collate_fn)

if __name__ == '__main__':
    main()
