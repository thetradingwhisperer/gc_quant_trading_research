FROM Python:3.10
# Path: requirements.txt
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app
