
import torch
import torch.nn as nn
from bert_seq2seq.basic_bert import BasicGLM
from bert_seq2seq.model.glm_model import GLMModel, GLMLargeChConfig
import os

class GLMSeq2SeqModel(BasicGLM):
    """
    """
    def __init__(self,
                 size="base", **kwargs):
        super(GLMSeq2SeqModel, self).__init__()
        if size == "base":
            pass
            print("不支持GLM base模型")
            os._exit(0)
        elif size == "large":
            config = GLMLargeChConfig()

        else :
            print("不支持的size")
            os._exit(0)

        self.config = config
        self.model = GLMModel(num_layers=config.num_layers,
                              vocab_size=config.vocab_size,
                              hidden_size=config.hidden_size,
                              num_attention_heads=config.num_attention_heads,
                              embedding_dropout_prob=config.embedding_dropout_prob,
                              attention_dropout_prob=config.attention_dropout_prob,
                              output_dropout_prob=config.output_dropout_prob,
                              max_sequence_length=config.max_sequence_length,
                              max_memory_length=config.max_memory_length,
                              checkpoint_activations=config.checkpoint_activations,
                              checkpoint_num_layers=config.checkpoint_num_layers,
                              output_predict=config.output_predict,
                              parallel_output=config.parallel_output,
                              relative_encoding=config.relative_encoding,
                              block_position_encoding=config.block_position_encoding,
                              spell_length=config.spell_length,
                              spell_func=config.spell_func,
                              attention_scale=config.attention_scale)

        self.hidden_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

    def forward(self, **data):
        input_ids = data["input_ids"]
        labels = data.get("labels", None)
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        return_memory = data.get("return_memory", False)
        loss_mask = data.get("loss_mask", None)
        mems = data.get("mems", None)

        if labels is not None:
            predictions = self.model(input_ids=input_ids, position_ids=position_ids,
                                     attention_mask=attention_mask, labels=labels,
                                     return_memory=return_memory, mems=mems, loss_mask=loss_mask)
            logits = predictions['logits']
            result_data = {"logits": logits, "hidden_states": predictions['hidden_states']}

            loss = predictions["loss"]
            result_data["loss"] = loss

        else :
            predictions = self.model(input_ids=input_ids, position_ids=position_ids,
                                     attention_mask=attention_mask,
                                     return_memory=return_memory, mems=mems)
            logits = predictions['logits']
            result_data = {"logits": logits, "hidden_states": predictions['hidden_states']}

        return result_data

    def load_weights(self, checkpoints_path):
        self.model.load_weights_glm(checkpoints_path)


