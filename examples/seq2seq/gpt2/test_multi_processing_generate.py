from cgi import test
from multiprocessing import Pool, Process
import os
from nbformat import write
import pandas as pd
import torch
from torch.utils.data import Dataset
from bert_seq2seq import Tokenizer
from bert_seq2seq import load_model
from bert_seq2seq import Predictor

vocab_path = "../state_dict/gpt2/vocab.txt"
model_save_path = "./gpt2_writing_model.bin" # 训练好的模型保存位置。

model_name = "gpt2"  # 选择模型名字
task_name = "seq2seq" # 任务名字

data_path = "../data/xzwaz2kx4cu.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer(vocab_path)
model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
model.load_all_params(model_save_path)
model.to(device)

predictor = Predictor(model, tokenizer)

def read_file():
    ## 更换数据集只需要实现这个函数即可，返回框架需要的src、tgt
    data = []

    df = pd.read_csv(data_path)
    for index, row in df.iterrows():
        if type(row[0]) is str:
            data.append(row[0])
    
    return data 

test_data = read_file()
print(f"data len is {len(test_data)}")

def generate_multiprocess(data):
    print(f"data is {data}")
    out = predictor.predict_generate_randomsample(data,
                                            input_max_length=100,
                                            out_max_length=900,
                                            top_k=50,
                                            top_p=0.8,
                                            repetition_penalty=3.0,
                                            temperature=1.5)

    with open(os.path.join("./gene", f"{data}.txt"), "w+") as f :
        f.write(str(out))
    # return (out, data)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    p = Pool(3)
    p.map_async(generate_multiprocess, test_data, chunksize=3)
    p.close()
    p.join()
    print('done.')