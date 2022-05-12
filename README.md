# bert_seq2seq_DDP
bert_seq2seq的DDP(分布式训练)版本。
此项目是对bert_seq2seq项目的重构并且很好的支持pytorch的DDP多卡训练。examples里面是各种训练例子，data中是样例数据。

本项目可以轻松调用不同种类transformer结构的模型（Bert、Roberta、T5、Nezha、Bart等）针对不同的任务（生成、序列标注、文本分类、关系抽取、命名实体识别等）进行快速的训练、预测，并且无缝进行分布式（DDP）训练。

**一个不同的数据集，只需要花5-10分钟修改好构建输入输出的函数，即可快速开始训练！**
#### 欢迎加入交流群～ 可以提问题，提建议，互相交流，还会提供部分数据与模型的下载 QQ群: 975907202 微信群: w11267191 加好友拉入群～


更多关于bert_seq2seq相关的内容请看：https://github.com/920232796/bert_seq2seq

### 项目特点一：
单卡训练与多卡训练方式相同，无需添加额外代码和使用额外命令运行。

单卡与多卡的运行方式均为：
```shell
python "./train.py" ## train.py为example中以train开头的训练脚本文件
```
切换多卡训练只需要修改 ```train.py``` 文件中的环境设置即可：

```python
num_gpus = 4 # gpu个数
num_nodes = 1 ## 机器个数 目前只支持1 ，多机待测试。
trainer = Trainer(env_type="DDP",## DDP为pytorch的分布式数据并行训练
                  epoches=5, model_save_dir=model_save_dir,
                  val_every_step=500, device=device,
                  batch_size=16, num_gpus=num_gpus, num_nodes=num_nodes,
                  training_script=__file__,
                  )
```
具体例子代码可以参考:

[train_roberta_auto_title_multi_gpu.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/train_roberta_auto_title_multi_gpu.py) 自动标题任务，多gpu训练。

### 项目特点二:
虽然使用Trainer类进行了封装，也能做到比较灵活的evaluation.

#### 自定义Evaluator类，可以自由进行验证（

[train_roberta_auto_title.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/train_roberta_auto_title.py) 自动标题任务，自定义validate函数。
[train_roberta_semantic_matching.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/train_roberta_semantic_matching.py) 语义匹配任务，自定义compute_metric函数。

### 项目特点三:
提供样例数据在data目录中，帮助理解代码运行过程，供参考（qq群文件里提供部分任务的全部数据）。
### 环境配置
#### 安装pytorch，不是太旧的版本即可。
https://pytorch.org/
#### 安装额外的包
```commandline
pip install bert_seq2seq_DDP 
pip install tqdm
pip install scikit-learn //可选
```
网络不好请切换国内源进行安装

#### 模型预训练参数、字典下载
1. roberta模型(支持base、large)，模型和字典文件下载地址：https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view 这里下载。 参考github仓库：https://github.com/ymcui/Chinese-BERT-wwm ，roberta-large模型也是在里面进行下载即可。
2. bert模型(支持base、large)，下载bert中文预训练权重 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin", 下载bert中文字典 "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt".
3. nezha模型，字典权重位置（支持base、large）：nezha-base模型下载：链接: https://pan.baidu.com/s/1Z0SJbISsKzAgs0lT9hFyZQ 提取码: 4awe
4. gpt2中文模型：gpt2中文通用模型和字典下载地址：https://pan.baidu.com/s/1vTYc8fJUmlQrre5p0JRelw  密码: f5un，下载好即可在 [examples/seq2seq/gpt2_text_writting.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/gpt2_text_writting.py) 中进行续写测试。
6. t5中文模型(支持base、small)，预训练参数下载：https://github.com/renmada/t5-pegasus-pytorch
7. SimBert模型，相似句的生成，预训练模型使用bert、roberta、nezha均可。
8. bart中文模型下载地址：https://huggingface.co/fnlp/bart-base-chinese

#### 参数说明，以文本分类任务为例
```python
target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"] # 所有labels
train_path = "../data/新闻标题文本分类/Train.txt" # 数据存放位置
model_name = "roberta" # 选择模型名字
task_name = "cls" # 任务名字
vocab_path = "../state_dict/roberta-large/vocab.txt" # roberta模型字典的位置
model_path = "../state_dict/roberta-large/pytorch_model.bin" # roberta模型位置
model_save_dir = "./state_dict/roberta_large_news_title_classification/" ## 训练结果和模型的保存位置
batch_size = 16
lr = 1e-5
# 加载字典
tokenizer = Tokenizer(vocab_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载roberta large 模型，做利用cls向量分类的任务。
bert_model = load_model(tokenizer.vocab,
                        model_name=model_name,
                        size="large", ## load large model
                        task_name=task_name,
                        target_size=len(target))
## 加载预训练的模型参数～
bert_model.load_pretrain_params(model_path)
# trainer设置
trainer = Trainer(epoches=3, val_every_step=500,  # 每500步进行验证和模型保存
                  batch_size=batch_size, 
                  env_type="pytorch", # 单卡训练方式
                  device=device, 
                  model_save_dir=model_save_dir, # 结果和模型的保存位置。
                  )
```
#### 运行
确定要做哪个任务，找到examples中对应的train_*.py文件，下载好模型与字典后，理解数据构建过程，运行即可（样例数据在data目录中，帮助理解代码过程，供参考）。

