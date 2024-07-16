import os
import requests
import time

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical advisor."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = LLMChain(llm=llm, 
                prompt=prompt,
                verbose=False,
                memory=memory)

def handle_response(user_input, history, custom_prompt):
    result = chain.invoke({"input": user_input})
    history.append((user_input, result["text"]))
    return result["text"]

chatbot = gr.ChatInterface(
                fn=handle_response,
                title="Sample Chatbot",
)

chatbot.launch()