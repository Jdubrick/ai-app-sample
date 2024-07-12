import os
import requests
import time

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain import OpenAI
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

llm = OpenAI(base_url=model_service,
                 model=model_name,
                 api_key="EMPTY",
                 streaming=True)

# Define a function to generate chatbot responses
def chatbot_response(user_input):
    response = llm.complete(prompt=user_input, max_tokens=50)
    return response['choices'][0]['text'].strip()

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="LangChain Chatbot",
    description="A simple chatbot using LangChain and Gradio",
    examples=[["Hello!"], ["What's the weather like?"], ["Tell me a joke."]]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
