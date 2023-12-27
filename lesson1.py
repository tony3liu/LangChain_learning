#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/12/5
# @Author  : 
# @File    : lesson1

import openai
from langchain.chat_models import ChatOpenAI

openai_key = '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'
# openai.api_key = os.environ["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"]
org_id = 'org-r9vpf3PYe1jlePpZOQy8xCle'
openai.api_key = openai_key
openai.organization = org_id

# 不用langchain自己写一个接口调用方法
# def get_openai_answer(prompt, model='gpt-3.5-turbo-0301'):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0
#     )
#     return response['choices'][0].message['content']
#
#
# res = get_openai_answer('hello,world？')
# print(res)

# 用langchain调用大模型接口
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain

# ======第一步：直接对LLM进行请求
llm = OpenAI(temperature=0.9)
text = "how are you"
answer_1 = llm(text)
# ======第二步：新建一个prompt_template,将用户输入对内容通过模版格式化
basic_prompt = PromptTemplate(
    input_variables=['product'],
    template='这是一个prompt模版{product}',
)
# prompt_using = prompt.format(product="用户输入的内容")
# ======第三步：将prompt_template和LLM组合
chain = LLMChain(llm=llm, prompt=basic_prompt)
chain_answer = chain.run(product='hello world')


# 同样我们还可以把这个prompt模版引入到聊天模型里
# 由于要用尽量人性化的语言对话，所以要把模版用HumanMessagePromptTemplate制作
human_message_prompt = HumanMessagePromptTemplate(
    input_variables=['example'],
    template='这是一个prompt模版{example}',
)
# 将基于人性化语言做的prompt模版，再套一层聊天的模版ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0)  # 定义使用的模型
chain_2 = LLMChain(llm=chat, prompt=chat_prompt)
# 使用
print(chain_2.run('hello world'))