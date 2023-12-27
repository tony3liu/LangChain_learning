#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/12/27
# @Author  : LiuTao
# @File    : embedding_model_test
from datetime import datetime
import json
import os
import time

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def create_result_file(file_name):
    # 获取当前脚本文件所在的目录路径
    folder = os.path.dirname(os.path.abspath(__file__))

    # 使用os.path.join构造完整文件路径
    file_path = os.path.join(folder, file_name)

    # 使用os.makedirs确保文件所在目录的存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("This is a result file."+"\n"+f"@Time:{formatted_time}"+"\n")
        print(f"File '{file_name}' created successfully at '{folder}'.")
    else:
        print(f"File '{file_name}' already exists at '{folder}'.")

    return file_path


def write_result_file(path, content):
    with open(path, 'a') as f:
        f.write(content)


def read_json(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def tst_embedding_score(model_name:str):
    current_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f'{current_path}/dataset.json'
    dataset = read_json(dataset_path)
    # print(dataset)
    create_result_file(f'result_{model_name}.txt')
    input_texts_list = dataset['input_text']
    for i in input_texts_list:
        input_texts = i
        print(input_texts)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        tolist = scores.tolist()
        print(tolist)
        content = f"{input_texts}:"+"\n"+f"{tolist}"
        write_result_file(f"{current_path}/result_{model_name}.txt", content + "\n")


# sentenceTransformers计算余弦相似度
def tst_embedding_using_cos_sim(model_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f'{current_path}/dataset.json'
    dataset = read_json(dataset_path)
    create_result_file(f'result_cos_sim_{model_name}.txt')
    sentences_list = dataset['data']
    for i in sentences_list:
        sentences = i
        sentences_model = SentenceTransformer(model_name)
        sentences_embeddings = sentences_model.encode(sentences)
        cos_sim_score = cos_sim(sentences_embeddings[0], sentences_embeddings[1])
        print(cos_sim_score)
        content_data = f"{sentences}:"+"\n"+f"{cos_sim_score}"
        write_result_file(f"{current_path}/result_cos_sim_{model_name}.txt", content_data + "\n")


if __name__ == "__main__":
    tst_embedding_score("thenlper/gte-base")
    tst_embedding_using_cos_sim("thenlper/gte-base")
