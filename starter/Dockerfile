FROM python:3.11-slim

WORKDIR /workspace/starter

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "src.run"]
