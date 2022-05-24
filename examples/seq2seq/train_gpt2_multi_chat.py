import os

import torch
from torch.utils.data import Dataset
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Trainer
from bert_seq2seq.dataset import gpt_collate_fn
from bert_seq2seq import Predictor
import json 

model_name = "gpt2"  # 选择模型名字
task_name = "seq2seq" # 任务名字

model_path = "../state_dict/gpt2/pytorch_model.bin"
vocab_path = "../state_dict/gpt2/vocab.txt"
model_save_path = "./gpt2_multi_chat_model.bin" # 训练好的模型保存位置。
lr = 2e-5
maxlen = 1024
data_path = '../data/LCCC-base-split/LCCC-base_train.json' # 数据位置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer(vocab_path)
model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
model.load_pretrain_params(model_path)
predictor = Predictor(model, tokenizer)

trainer = Trainer(env_type="pytorch", 
                  epoches=5,
                  val_every_step=500,
                  device=device, 
                  batch_size=8,
                  gradient_accmulation_step=8)

def read_file():
    ## 更换数据集只需要实现这个函数即可，返回框架需要的src
    
    with open(data_path) as f:
        data = json.loads(f.read())

    return data 

data = read_file()
print(data[:5])

class ChatDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data) :
        ## 一般init函数是加载所有数据
        super().__init__()
        self.data = data

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        data = self.data[i]
        input_ids = [tokenizer.token_start_id]

        for index, text in enumerate(data):
            if (index + 1) % 2 == 1:
                text = "A:" + text
            else :
                text =  "B:" + text

            text_ids = tokenizer.encode_plus(text, max_length=maxlen)["input_ids"][1:]
            input_ids.extend(text_ids)

        output = {
            "input_ids": input_ids,
        }
        return output

    def __len__(self):

        return len(self.data)

class Evaluator:

    def on_validation(self):
        ## 自己定义validate函数实现，十分灵活。
        test_data = [["A:今天天气很好，你觉得呢？"],
                     ["A:我去吃了火锅。"],
                     ["A:我去吃了火锅。", "B:我也是，真不错，你吃的哪家？"]
                     ]
        for text in test_data:
            print(predictor.predict_multi_response(text,
                                                  input_max_length=200,
                                                  out_max_length=40,
                                                  top_k=30, top_p=0.9,
                                                  repetition_penalty=1.2,
                                                  temperature=1.2))

        torch.save(model.state_dict(), model_save_path)
        print(f"模型保存成功～")

def main():
    ## 加载数据
    data = read_file()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    train_dataset = ChatDataset(data)

    trainer.train(model, optimizer, train_dataset=train_dataset, evaluator=Evaluator,
                  collate_fn=gpt_collate_fn)

if __name__ == '__main__':
    main()
