FROM registry.access.redhat.com/ubi9/python-311:1-62.1716478620
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY chatbot_ui.py .
EXPOSE 7860
ENTRYPOINT [ "python", "sample-app.py" ]