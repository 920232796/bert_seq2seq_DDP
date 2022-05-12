import os
import torch
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Predictor


model_name = "roberta" # 选择模型名字
task_name = "cls"
vocab_path = "../state_dict/roberta/vocab.txt" # roberta模型字典的位置
model_save_path = "./bert_emotion_analysis.bin"
# 加载字典
tokenizer = Tokenizer(vocab_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = ["中性", "积极", "消极"]

def main():
    bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name, target_size=3)
    bert_model.load_all_params(model_save_path)
    predictor = Predictor(bert_model, tokenizer)

    text = ["今天天气很好，挺喜欢。",
            "你今天是生谁的气了？怎么这么不开心？？",
            "明天要下雨了。"]

    for t in text:
        ids = predictor.predict_cls_classifier(t).argmax(dim=1)
        print(target[ids])

if __name__ == '__main__':
    main()
