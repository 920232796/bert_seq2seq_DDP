
import torch
from bert_seq2seq.model.t5_model import T5ForConditionalGeneration, T5Config, T5SmallConfig
from bert_seq2seq.basic_bert import BasicT5
import torch.nn.functional as F

class T5Model(BasicT5):

    def __init__(self, size="base"):
        super().__init__()
        if size == "base":
            config = T5Config()
        elif size == "small":
            config = T5SmallConfig()
        else:
            raise Exception("not support this model type")
        self.model = T5ForConditionalGeneration(config)

    def forward(self, **data):
        input_ids = data["input_ids"]
        decoder_input_ids = data["decode_input_ids"]
        labels = data.get("labels", None)
        t5_out = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        if labels is not None:
            return {"logits": t5_out[1], "loss": t5_out[0]}

        return {"logits": t5_out[0]}