# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from bert_seq2seq import Predictor
from bert_seq2seq import GLMTokenizer
from bert_seq2seq.utils import load_model
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    tokenizer = GLMTokenizer("../state_dict/GLM-large-ch/cog-pretrain.model")
    model = load_model(model_name="glm",
                       task_name="seq2seq",
                       size="large")

    model.load_pretrain_params("../state_dict/GLM-large-ch/pytorch_model.bin")
    model.to(device)

    predictor = Predictor(model, tokenizer)
    # generate samples
    text = [
        '问题：啤酒伤胃吗？回答：[gMASK]', "问题：隔夜菜能吃吗？回答：[gMASK]", "问题：如何评价许嵩？回答：[gMASK]"
    ]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)

    text = ['北京故宫是中国[MASK]非物质文化遗产。', "上海是中国[MASK]大都市。", "天津大学是[MASK]现代大学。"]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)
    #
    text = [
        "人工智能是一个以计算机科学为基础，由计算机、数学、哲学等多学科交叉融合的交叉学科，[sMASK]，具有非常巨大的前景。",
        "最近十多年来，人工神经网络的研究工作不断深入，已经取得了很大的进展，[sMASK]，表现出了良好的智能特性。"
    ]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)
