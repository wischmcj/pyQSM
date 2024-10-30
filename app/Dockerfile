
FROM python:3.9

ARG ssh_prv_key
ARG ssh_pub_key

RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \ 
    libdrm2 \
    libedit2 \
    libexpat1 \
    libgcc-s1 \ 
    libglapi-mesa \
    libx11-xcb1 \
    libxcb-dri2-0 libxcb-glx0 libxcb-shm0 libxcb-xfixes0 libxfixes3 \
    libxxf86vm1 \
    python3-pip \
    gfortran \
    build-essential \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# copying ssh keys to configure 
#  ssh connection
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh

# Copying known hosts to allow my main 
#  pc (client) to ssh into the docker container 
#  just as it would the pc running this container (server)
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
COPY ./known_hosts  /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# Avoid cache purge by adding requirements first
ADD ./requirements.txt /code/requirements.txt
# COPY requirements.txt requirements.txt

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade open3d

# Setup env
WORKDIR /code
#ENV FLASK_APP app.py
#ENV FLASK_RUN_HOST 0.0.0.0
#ENV RUN_ENVIRONMENT docker
ENV WEBRTC_IP=0.0.0.0
ENV WEBRTC_PORT=5000

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

# Remove SSH keys for security reasons
RUN rm -rf /root/.ssh/

# Copy code over and run 
# ADD . .
COPY / /code
#CMD ["flask", "run"]
CMD ["python3","test_server.py"]
