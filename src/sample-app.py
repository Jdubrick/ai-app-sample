import os
import requests
import time

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


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

memory = ConversationBufferWindowMemory(return_messages=True,k=4) # Store prior 4 messages


llm = ChatOpenAI(base_url=model_service,
                 model=model_name,
                 api_key="no-key",
                 )

template = """You are the best advisor in the world.

A user has come to you and asks the following question:
{message}

If you need to ask clarifying questions you may do so.
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm
# chain = LLMChain(llm=llm, 
#                 prompt=prompt,
#                 verbose=False,
#                 memory=memory)

def handle_response(message, history):
    
    conversation = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])
    print(f"CONVERSATION: {conversation}")
    result = chain.invoke({
            "message": message
        }
    )

    print(f"RESULT: {result}")
    output_resp = ""
    for char in result.content:
        output_resp += char
        time.sleep(0.03)
        yield output_resp
    # return result.content


chatbot = gr.ChatInterface(
                fn=handle_response,
                title="Sample Chatbot",
)

chatbot.launch(server_port=8501)