import gradio as gr
import openai
import json
import pandas as pd
import glob

# langchain 相关
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

# os
import os

# 获取Api Key

def get_api_key():
    # 可以自己根据自己实际情况实现
    # 以我为例子，我是存在一个 openai_key 文件里，json 格式
    '''
    {"api": "keyxxx"}
    '''
    openai_key_file = 'api.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api']

openai.api_key = get_api_key()
os.environ["OPENAI_API_KEY"] = openai.api_key

# text文件处理函数
# Create a gr.State() object to store the generated textindex
# text_index = gr.State("")

# 文本检查
def combine_text(files):
    # 文件路径
    multitext=[]
    paths=[]
    # read the content of each file as a string
    for file in files:
        #with open(path) as f:
        path=file.name # 等于是需要多来一步
        paths.append(path)
        with open(path) as f:
            multitext.append(f.read())
        # append the content to the combined string
    return multitext

# openai对话处理函数
def openaichat(ques):
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",messages=[
            {"role": "user", "content": ques}
            ]
    )
    reponse=rsp.get("choices")[0]["message"]["content"]
    return reponse

def docchat(text_index,ques): # 先做基于一个文档的
    index = text_index
    docreponse=index.query(ques)
    return docreponse

# gradio 组件
with gr.Blocks() as demo:
    # 空容器
    text_indexs = gr.State()
    textrow = gr.State()
    # 文件上传
    with gr.Tab("文件上传"):
        gr.Markdown(
        """
        # 上传你的文件
        """)
        # 上传文件内容展示
        file_input = gr.File(type="file",file_count="multiple")
        text_output = gr.Textbox()

        b1 = gr.Button("检查文本") # 组件

        b1.click(combine_text,
        inputs=file_input,
        outputs=text_output) # 组件行为
    
    # 非基于上传文件的对话
    with gr.Tab("chat with GPT3.5"):
        gr.Markdown(
        """
        # Chat with GPT3.5-turbo
        """)
        
        inp = gr.Textbox(label="Your Question",placeholder="What is your question?")
        out = gr.Textbox(label="AI Answer")
        
        greet_btn = gr.Button("ASK NOW!")
        
        greet_btn.click(fn=openaichat, inputs=inp, outputs=out, api_name="openaichat")

    # 基于单个文件的对话
    with gr.Tab("Chat doc through GPT3.5"):
        gr.Markdown(
        """
        # Chat doc through GPT3.5
        # 上传你的文档并开始对话
        """)

        # 文本编码
        def text_indexer(files):
            path=files[0].name

            # 使用容器 检查文本
            with open(path) as f:
                multitext=f.read()

            # 使用一个容器 索引化文本
            loader = TextLoader(path)
            text_index = VectorstoreIndexCreator().from_loaders([loader])
            return {textrow: multitext,text_indexs: text_index}

        # 文本index化
        file_input = gr.File(type="file",file_count="multiple")
        textrow = gr.Textbox()
        b1 = gr.Button("文本index化") # 组件
        b1.click(text_indexer,
            inputs=file_input,
            outputs=[textrow,text_indexs],
            api_name="text_indexer") # 组件行为
        
        # 会话
        inp = gr.Textbox(label="Your Question",placeholder="What is your question?")
        out = gr.Textbox(label="AI Answer")
        greet_btn = gr.Button("ASK NOW!")        
        greet_btn.click(fn=docchat, inputs=[text_indexs,inp],
        outputs=out,
        api_name="docchat")
    
demo.launch()