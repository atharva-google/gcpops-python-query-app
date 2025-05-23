FROM python:3.10
EXPOSE 8080
WORKDIR /gcpops-python-query-app
COPY . ./

RUN pip install -r requirements.txt
RUN python create_chroma.py

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]