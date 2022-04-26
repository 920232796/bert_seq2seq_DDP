import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np

def padding(indice, max_length, pad_idx=0):
    """
    pad 函数
    """

    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)

def gpt_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded.clone()
    target_ids_padded[target_ids_padded == 0] = -100

    return {
        "input_ids": token_ids_padded,
        "labels": target_ids_padded
    }

def bert_seq2seq_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": token_type_ids_padded,
        "labels": target_ids_padded
    }

def bert_cls_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["labels"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": token_type_ids_padded,
        "labels": target_ids
    }

def bert_sequence_label_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    token_ids = [data["input_ids"] for data in batch]
    
    max_length = max([len(t) for t in token_ids])
    target_ids = [data["labels"] for data in batch]
  
    token_ids_padded = padding(token_ids, max_length)

    target_ids_padded = padding(target_ids, max_length)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": None,
        "labels": target_ids_padded
    }

def bert_sequence_label_gp_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
    token_ids = [data["input_ids"] for data in batch]
    labels = [data["labels"] for data in batch]
    token_ids_padded = sequence_padding(token_ids)
    labels_padded = sequence_padding(labels, seq_dims=3)
    token_ids_padded = torch.from_numpy(token_ids_padded)
    labels_padded = torch.from_numpy(labels_padded)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": None,
        "labels": labels_padded
    }

ALL_COLLATE = {
    "gpt2": gpt_collate_fn,
    "bert_seq2seq": bert_seq2seq_collate_fn,
    "bert_cls": bert_cls_collate_fn,
    "bert_sequence_labling": bert_sequence_label_collate_fn,
    "sequence_labeling_crf": bert_sequence_label_collate_fn,
    "bert_multilabel_cls": bert_cls_collate_fn,

}

class AbstractDataset(Dataset):

    def __init__(self, model_name, model_class) -> None:
        super().__init__()


        self.collate_fn = ALL_COLLATE.get(f"{model_name}_{model_class}", None)
        if self.collate_fn is None:
            import os
            print("illegal model_name or model_class")
            os._exit(0)

    
    def __getitem__(self, index):
        return NotImplemented
    
    def __len__(self):
        return NotImplemented