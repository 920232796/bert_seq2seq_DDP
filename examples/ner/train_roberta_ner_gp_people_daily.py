# 人民日报数据
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq.dataset import bert_sequence_label_gp_collate_fn
from bert_seq2seq import Trainer
import numpy as np
import os
from bert_seq2seq import Predictor

train_path = "../data/china-people-daily-ner-corpus/example.train"
valid_path = '../data/china-people-daily-ner-corpus/example.dev'
test_path = '../data/china-people-daily-ner-corpus/example.test'

model_name = "roberta" # 选择模型名字
task_name = "sequence_labeling_gp"

vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta/pytorch_model.bin" # roberta模型位置

model_save_path = "./bert_sequence_labeling_gp.bin"

batch_size = 16
lr = 2e-5
# 加载字典
tokenizer = Tokenizer(vocab_path)
maxlen = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(epoches=10,
                  env_type="pytorch",
                  val_every_step=500,
                  batch_size=batch_size,
                  device=device,
                  )
target = set()

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    target.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i

            D.append(d)
    return D

train_data = load_data(train_path)
val_data = load_data(valid_path)
test_data = load_data(test_path)
target = list(sorted(target))

bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name,
                        target_size=len(target), ner_inner_dim=64)
# ## 加载预训练的模型参数～
bert_model.load_pretrain_params(model_path)
predictor = Predictor(bert_model, tokenizer)

## 自定义dataset
class NERDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data) :
        ## 一般init函数是加载所有数据
        super(NERDataset, self).__init__()
        # 读原始数据
        self.data = data
    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        data = self.data[i]

        tokens = tokenizer.tokenize(data[0], maxlen=maxlen, add_spatial_tokens=True)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        mapping = tokenizer.rematch(data[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        length = len(tokens)
        labels = np.zeros((len(target), length, length))

        for start, end, label in data[1:]:
            if start in start_mapping and end in end_mapping:
                # 说明找到这个token了。
                start = start_mapping[start]
                end = end_mapping[end]

                label_index = target.index(label)
                labels[label_index, start, end] = 1

        output = {
            "input_ids": input_ids,
            "labels": labels
        }
        return output

    def __len__(self):
        return len(self.data)


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(predictor.predict_ner(d[0], target, maxlen=maxlen))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator:

    def __init__(self):
        self.best_val_f1 = 0.0

    def on_validation(self, data):
        loss = data["loss"]
        step = data["iteration"]

        text = ["6月15日，河南省文物考古研究所曹操高陵文物队公开发表声明承认：“从来没有说过出土的珠子是墓主人的",
                "4月8日，北京冬奥会、冬残奥会总结表彰大会在人民大会堂隆重举行。习近平总书记出席大会并发表重要讲话。在讲话中，总书记充分肯定了北京冬奥会、冬残奥会取得的优异成绩，全面回顾了7年筹办备赛的不凡历程，深入总结了筹备举办北京冬奥会、冬残奥会的宝贵经验，深刻阐释了北京冬奥精神，对运用好冬奥遗产推动高质量发展提出明确要求。",
                "当地时间8日，欧盟委员会表示，欧盟各成员国政府现已冻结共计约300亿欧元与俄罗斯寡头及其他被制裁的俄方人员有关的资产。",
                "这一盘口状态下英国必发公司亚洲盘交易数据显示博洛尼亚热。而从欧赔投注看，也是主队热。巴勒莫两连败，",
                ]
        for t in text:
            entities = predictor.predict_ner(t, target, maxlen=maxlen)
            result = {}
            for e in entities:
                if e[2] not in result:
                    result[e[2]] = [t[e[0]: e[1]+1]]
                else :
                    result[e[2]].append(t[e[0]: e[1]+1])
            print(f"result is {result}")

        f1, precision, recall = evaluate(val_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            torch.save(bert_model.state_dict(), model_save_path)
            print(f"模型保存成功～")

        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


def main():

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=lr, weight_decay=1e-5)
    train_dataset = NERDataset(train_data)
    trainer.train(model=bert_model, optimizer=optimizer, evaluator=Evaluator,
                  train_dataset=train_dataset, collate_fn=bert_sequence_label_gp_collate_fn,
                  )

if __name__ == '__main__':
    main()
