FROM python:3.10

WORKDIR /usr/project
COPY . .
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python", "./notebooks/demo.py"]

