## 例子文件说明

### bert embedding
bert、roberta、nezha模型，输入一个句子，得到这个句子的embedding
1. [get_bert_embedding.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/bert_embedding/get_bert_embedding.py)

### ner 
bert、roberta、nezha模型，命名实体识别任务，支持crf与global pointer方式
1. [train_bert_ner_crf_people_daily.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/ner/train_bert_ner_crf_people_daily.py)  crf方式进行ner任务
2. [train_roberta_ner_gp_people_daily.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/ner/train_roberta_ner_gp_people_daily.py)  global pointer 方式进行ner任务

### seq2seq
生成任务，支持bert、roberta、nezha、gpt2、t5、bart等模型
1. [test_gpt2_text_writting.py.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/test_gpt2_text_writting.py.py) gpt2续写测试
2. [train_roberta_auto_title.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/train_roberta_auto_title.py) roberta模型自动标题训练
3. [train_roberta_auto_title_multi_gpu.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/train_roberta_auto_title_multi_gpu.py)  roberta自动标题训练（多gpu版本）
4. [train_gpt2_multi_chat.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/train_gpt2_multi_chat.py)  gpt2多轮对话训练
5. [test_t5_auto_title.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/test_t5_auto_title.py)  T5模型自动标题测试代码
6. [test_gpt2_multi_chat.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/seq2seq/test_gpt2_multi_chat.py)  gpt2多轮对话测试

### text classification
bert、roberta、nezha模型，支持文本分类、情感分析、语义匹配任务
1. [train_roberta_news_title_classification.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/train_roberta_news_title_classification.py) 新闻摘要文本分类训练
2. [train_roberta_news_title_classification_multi_gpu.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/train_roberta_news_title_classification_multi_gpu.py) 新闻摘要文本分类训练（多gpu版本）
3. [train_roberta_semantic_matching.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/train_roberta_semantic_matching.py) 语义匹配训练
4. [test.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/test.py) 加载训练好的模型进行测试

### FAQ 检索式问答
1. [1_construct_data.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/1_construct_data.py) 构建数据集，提前提取embedding特征
2. [2_test_bert_faq.py](https://github.com/920232796/bert_seq2seq_DDP/blob/master/examples/text_classification/2_test_bert_faq.py) 加载构建的embeddings，利用faiss进行相似问题的检索
