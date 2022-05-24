from bert_seq2seq.utils import load_model
from bert_seq2seq.tokenizer import T5PegasusTokenizer
from bert_seq2seq import Predictor

model_path = "../state_dict/t5-chinese/pytorch_model.bin"
vocab_path = "../state_dict/t5-chinese/vocab.txt"

tokenizer = T5PegasusTokenizer(vocab_path)

model = load_model(tokenizer.vocab,
                   model_name="t5",
                   task_name="seq2seq")
model.load_pretrain_params(model_path)

predictor = Predictor(model, tokenizer)

if __name__ == '__main__':
    text = "本文总结了十个可穿戴产品的设计原则，而这些原则同样也是笔者认为是这个行业最吸引人的地方：1.为人们解决重复性问题，2.从人开始而不是从机器开始，3.要引起注意但不要刻意，4.提升用户能力而不是取代人"
    out = predictor.predict_generate_randomsample(text, out_max_length=100,
                                                  repetition_penalty=1.0,
                                                  top_p=0.9, top_k=50)
    print(out)