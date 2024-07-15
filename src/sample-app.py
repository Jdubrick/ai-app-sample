import os
import requests
import time

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage

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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Translate all questions to the best of your ability from English to German.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm

# chain = LLMChain(llm=llm, prompt=prompt)

# Define a function to generate chatbot responses
def chatbot_response(user_input):
    res = chain.invoke(
        {"messages": [HumanMessage(content=f"{user_input}")]}
    )
    return res.content
    

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="Translator Bot",
    description="A simple chatbot using LangChain and Gradio",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
