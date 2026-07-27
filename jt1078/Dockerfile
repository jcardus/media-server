FROM python:3.12-alpine

RUN apk add --no-cache ffmpeg

WORKDIR /app
COPY main.py .

CMD ["python", "-u", "main.py"]
