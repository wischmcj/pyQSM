# app/Dockerfile

FROM python:3.9-slim

WORKDIR /stl

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    xvfb \
    libegl1 \
    libgl1 \
    libgomp1 \
    libxrender1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/streamlit/streamlit-example.git .
COPY . /stl

ENV DISPLAY=${DISPLAY}

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:5001/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=5001", "--server.address=0.0.0.0"]
