FROM python

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Run DVC pipeline during build
RUN dvc repro

# Start Flask server
CMD ["python", "app.py"]