import os

import torch
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import GLMTokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Trainer
from bert_seq2seq.dataset import glm_generation_collate_fn
from bert_seq2seq import Predictor

model_name = "glm"  # 选择模型名字
task_name = "seq2seq" # 任务名字

vocab_path = "../state_dict/GLM-large-ch/cog-pretrain.model"  # roberta模型字典的位置
model_path = "../state_dict/GLM-large-ch/pytorch_model.bin"  # 预训练模型位置
model_save_path = "./GLM_auto_title_model.bin" # 训练好的模型保存位置。

lr = 1e-5
maxlen=1024

src_dir = '../data/auto_title/train.src' # 数据位置
tgt_dir = '../data/auto_title/train.tgt'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = GLMTokenizer(vocab_path)

model = load_model(model_name=model_name,
                   task_name=task_name,
                   size="large")

model.load_pretrain_params(model_path)
predictor = Predictor(model, tokenizer)

trainer = Trainer(env_type="pytorch",
                  epoches=5,
                  val_every_step=500,
                  device=device,
                  batch_size=16)

def read_file():
    ## 更换数据集只需要实现这个函数即可，返回框架需要的src、tgt
    src = []
    tgt = []

    with open(src_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())

    return src,tgt

class AutoTitleDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super().__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]

        tokenizer_out = tokenizer.encode_plus(src,
                                              tgt,
                                              max_length=maxlen,
                                              # mask_token="sMASK",
                                              prefix_flag="标题生成：",
                                              post_flag=" 回答：")

        output = {
            "input_ids": tokenizer_out["input_ids"],
            "attention_mask": tokenizer_out["attention_mask"],
            "position_ids": tokenizer_out["position_ids"],
            "loss_mask": tokenizer_out["loss_mask"],
        }
        return output

    def __len__(self):

        return len(self.sents_src)

class Evaluator:

    def prompt(self, text):
        text = "标题生成：" + text + "回答：[gMASK]"
        return text

    def on_validation(self, data):
        loss = data["loss"]
        step = data["iteration"]
        ## 自己定义validate函数实现，十分灵活。
        test_data = ["本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
                     "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
                     "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"]
        for text in test_data:
            text = self.prompt(text)
            print(predictor.predict_generate_randomsample(text,
                                                          top_k=50,
                                                          top_p=0.9,
                                                          repetition_penalty=4.0,
                                                          input_max_length=600,
                                                          out_max_length=100,
                                                          ))

        torch.save(model.state_dict(), model_save_path)
        print(f"模型保存成功～")

def main():
    ## 加载数据
    all_src, all_tgt = read_file()
    train_size = int(len(all_src) * 0.9)
    train_src, train_tgt = all_src[:train_size], all_tgt[:train_size]
    # 声明需要优化的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    train_dataset = AutoTitleDataset(train_src, train_tgt)

    trainer.train(model, optimizer, train_dataset=train_dataset, evaluator=Evaluator,
                  collate_fn=glm_generation_collate_fn)

if __name__ == '__main__':
    main()
