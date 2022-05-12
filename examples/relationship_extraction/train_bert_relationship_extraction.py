
import json 
import numpy as np
import torch 
import os
from tqdm import tqdm 
from bert_seq2seq import Trainer
from bert_seq2seq import Tokenizer
from torch.utils.data import Dataset
from bert_seq2seq.dataset import bert_gplinker_collate_fn, sequence_padding
from bert_seq2seq.utils import load_model

vocab_path = "../state_dict/roberta/vocab.txt"
model_path = "../state_dict/roberta/pytorch_model.bin"
model_save_path = "./bert_relation_extraction.bin"
task_name = "relationship_extraction"
model_name = "roberta"
epoches = 5
data_dir = "../data/三元组抽取"
train_path = os.path.join(data_dir, "train_data.json")
val_path = os.path.join(data_dir, "dev_data.json")

batch_size = 8
maxlen = 128
lr = 1e-5
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
trainer = Trainer(epoches=epoches, env_type="pytorch",
                  val_every_step=1000, batch_size=batch_size,
                  device=device,
                  # num_nodes=1,
                  # num_gpus=4,
                  # training_script=__file__,
                  )

def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D

def load_target():
    target = []
    with open(os.path.join(data_dir, 'all_50_schemas')) as f:
        for l in f:
            l = json.loads(l)
            if l['predicate'] not in target:
                target.append(l['predicate'])
    return target

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

# 建立分词器
tokenizer = Tokenizer(vocab_path)
train_data = load_data(train_path)
valid_data = load_data(val_path)
target = load_target()
model = load_model(tokenizer.vocab, model_name=model_name, 
                    task_name=task_name, target_size=len(target), 
                    ner_inner_dim=64 )

class RelationshipDataset(Dataset):
    def __init__(self, data):
        pass 
        self.data = data 

    def __getitem__(self, i):
        data = self.data[i]

        tokenizer_out = tokenizer.encode_plus(data["text"], max_length=maxlen, truncation=True)
        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]

        spoes = set()
        for s, p, o in data['spo_list']:
            s = tokenizer.encode_plus(s)["input_ids"][1:-1]
            p = target.index(p)

            o = tokenizer.encode_plus(o)["input_ids"][1:-1]
            sh = search(s, input_ids)
            oh = search(o, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
        
        # 构建标签
        entity_labels = [set() for _ in range(2)]
        head_labels = [set() for _ in range(len(target))]
        tail_labels = [set() for _ in range(len(target))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st))
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))
            tail_labels[p].add((st, ot))

        for label in entity_labels + head_labels + tail_labels:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        output = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "entity_labels": entity_labels,
            "head_labels": head_labels,
            "tail_labels": tail_labels,
        }
        return output

    def __len__(self):
        return len(self.data)

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen, add_spatial_tokens=True)
    mapping = tokenizer.rematch(text, tokens)
    tokenizer_out = tokenizer.encode_plus(text, max_length=maxlen)
    input_ids = tokenizer_out["input_ids"]
    token_type_ids = tokenizer_out["token_type_ids"]
    
    input_ids = torch.tensor(input_ids, device=device)
    token_type_ids = torch.tensor(token_type_ids, device=device)
    if input_ids.ndim == 1:
        input_ids = input_ids.view(1, -1)
        token_type_ids = token_type_ids.view(1, -1)
    with torch.no_grad():
        model_out = model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})

    outputs = [model_out["entity_output"].cpu().numpy(), 
                model_out["head_output"].cpu().numpy(), 
                model_out["tail_output"].cpu().numpy()]

    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                try:
                    spoes.add((
                        text[mapping[sh][0]:mapping[st][-1] + 1], target[p],
                        text[mapping[oh][0]:mapping[ot][-1] + 1]
                    ))
                except:
                    continue

    return list(spoes)

def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')

    for d in tqdm(data, total=len(data)):
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    f.close()
    return f1, precision, recall

def validate():
    text = "南京京九思新能源有限公司于2015年05月15日在南京市江宁区市场监督管理局登记成立"
    spo_list = extract_spoes(text)
    print(f"spo_list is {spo_list}")
    f1, precision, recall = evaluate(valid_data)
    print(f"f1 is {f1}, precision is {precision}, recall is {recall}")
    return f1

class Evaluator:
    def __init__(self):
        self.best_f1 = 0.0

    def on_validation(self):
        pass

    def on_epoch_end(self):
        f1 = validate()
        if self.best_f1 > f1:
            torch.save(model, model_save_path)
            print(f"模型保存成功: {model_save_path}")


if __name__ == "__main__":
    
    train_dataset = RelationshipDataset(train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    trainer.train(model=model, optimizer=optimizer,
                  train_dataset=train_dataset,
                  evaluator=Evaluator,
                  collate_fn=bert_gplinker_collate_fn)