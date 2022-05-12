## 多轮对话，测试
from bert_seq2seq.utils import load_model
from bert_seq2seq.tokenizer import Tokenizer
from bert_seq2seq import Predictor
import os 

vocab_path = "../state_dict/gpt2/vocab.txt"
model_save_path = "./gpt2_multi_chat_model.bin" # 训练好的模型保存位置。

tokenizer = Tokenizer(vocab_path)

model = load_model(tokenizer.vocab,
                   model_name="gpt2",
                   task_name="seq2seq")
model.load_all_params(model_save_path)
predictor = Predictor(model, tokenizer)

if __name__ == '__main__':
    sentences_list = [["今天我去吃了火锅，还可以，想不想尝尝？"],
                      ["今天天气很好", "是啊，真的非常好，我也出去玩了一会"],
                      ["今天天气很好", "是啊，真的非常好", "你也出去玩了吗？"]]

    for sentences in sentences_list:
        out = predictor.predict_multi_response(sentences,
                                                repetition_penalty=1.2,
                                                temperature=1.2,
                                                top_p=1.0, top_k=30)
        print(out)