import torch
from torch.utils.data import Dataset
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq.dataset import bert_cls_collate_fn
from bert_seq2seq.trainer import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from bert_seq2seq import Predictor
import os
from tqdm import tqdm

target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
train_path = "../data/新闻标题文本分类/Train.txt"
model_name = "roberta" # 选择模型名字
task_name = "cls"
vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta/pytorch_model.bin" # roberta模型位置
model_save_path = "./bert_news_title_classification.bin"
batch_size = 16
lr = 1e-5
# 加载字典
tokenizer = Tokenizer(vocab_path)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

trainer = Trainer(epoches=3,
                  val_every_step=10000,
                  batch_size=16,
                  env_type="pytorch",
                  device=device)
bert_model = load_model(tokenizer.vocab,
                        model_name=model_name,
                        task_name=task_name,
                        target_size=len(target))
## 加载预训练的模型参数～
bert_model.load_pretrain_params(model_path)
predictor = Predictor(bert_model, tokenizer)

def read_corpus():
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []

    with open(train_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        sents_tgt.append(int(line[0]))
        sents_src.append(line[2])

    return sents_src, sents_tgt

## 加载数据
all_input, all_label = read_corpus()
train_input, val_input, train_label, val_label = train_test_split(all_input, all_label, train_size=0.8, random_state=123)


## 自定义dataset
class ClassificationDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        super(ClassificationDataset, self).__init__()
        # 读原始数据
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

    def __getitem__(self, i):
        ## 得到单个数据
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

def validate():
    res = []
    for data in tqdm(val_input, total=len(val_input)):
        pred = predictor.predict_cls_classifier(data)
        pred = pred.argmax(dim=0).numpy()
        res.append(pred)

    f1 = f1_score(val_label, res, average="macro")
    accuracy = accuracy_score(val_label, res)
    recall = recall_score(val_label, res, average="macro")
    precision = precision_score(val_label, res, average="macro")

    print(f" f1 is {f1}, acc is {accuracy}, recall is {recall} precision is {precision}")
    return accuracy

class Evaluator:
    def __init__(self):
        self.best_acc = 0.0

    def on_epoch_end(self):
        acc = validate()
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(bert_model, model_save_path)
            print(f"模型保存成功～")

    def on_validation(self):
        pass

def main():

    # 声明需要优化的参数
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=lr, weight_decay=1e-3)
    train_dataset = ClassificationDataset(train_input, train_label)

    trainer.train(bert_model, optimizer=optimizer,
                  train_dataset=train_dataset,
                  evaluator=Evaluator,
                  collate_fn=bert_cls_collate_fn)

if __name__ == '__main__':
    main()
