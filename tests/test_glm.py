from bert_seq2seq.glm_tokenizer import GLMTokenizer
from bert_seq2seq.utils import load_model

tokenizer = GLMTokenizer("cog-pretrain.model")

tokenizer_out = tokenizer.encode_plus("今天天气好", "适合出去玩")

print(f"ids is {tokenizer_out['input_ids']}")
print(f"attention mask is {tokenizer_out['attention_mask']}")
print(f"position ids is {tokenizer_out['position_ids']}")

out = tokenizer.decode(tokenizer_out["input_ids"])

print(out)

print(f"target ids is {tokenizer.decode(tokenizer_out['target_ids'])}")
