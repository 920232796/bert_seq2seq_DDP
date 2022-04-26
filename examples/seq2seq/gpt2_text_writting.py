from bert_seq2seq.utils import load_model
from bert_seq2seq.tokenizer import Tokenizer
from bert_seq2seq import Predictor

model_path = "../state_dict/gpt2/pytorch_model.bin"
vocab_path = "../state_dict/gpt2/vocab.txt"

tokenizer = Tokenizer(vocab_path)

model = load_model(tokenizer.vocab,
                   model_name="gpt2",
                   task_name="seq2seq")
model.load_pretrain_params(model_path)

predictor = Predictor(model, tokenizer)

if __name__ == '__main__':
    text = "今天天气好，"
    out = predictor.predict_generate_randomsample(text, out_max_length=100,
                                                  repetition_penalty=1.5,
                                                  top_p=1.0, top_k=20, seed=123)
    print(out)