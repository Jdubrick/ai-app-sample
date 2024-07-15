import os
import requests
import time

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage


model_endpoint = os.getenv("MODEL_ENDPOINT", "http://localhost:8001")
model_service = f"{model_endpoint}/v1"

def checking_model_service():
    start = time.time()
    print("Checking Model Service Availability...")
    ready = False
    while not ready:
        try:
            request = requests.get(f'{model_service}/models')
            if request.status_code == 200:
                ready = True
        except:
            pass
        time.sleep(1) 
    print("Model Service Available")
    print(f"{time.time()-start} seconds")

checking_model_service()
model_name = os.getenv("MODEL_NAME", "")

llm = ChatOpenAI(base_url=model_service,
                 model=model_name,
                 api_key="EMPTY",
                 max_tokens=None,
                 )

prompt = ChatPromptTemplate.from_template("Translate the sentence after the colon from English to German: {message}")
chain = prompt | llm

def llm_result(message):
    return chain.invoke({
        "message": message
    })

with gr.Blocks() as demo:
    gr.Markdown("Enter a sentence you wish to be translated and hit **Run** to see the output.")
    inpt = gr.Textbox(placeholder="Sentence to translate")
    outpt = gr.Textbox(placeholder="Translation ...", ) 
    btn = gr.Button("Run")
    btn.click(fn=llm_result, inputs=inpt, outputs=outpt)

demo.launch()