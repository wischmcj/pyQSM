
#ARG ssh_prv_key
#ARG ssh_pub_key

FROM ubuntu:20.04
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    libegl1 libgl1 libgomp1 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copying ssh keys to configure 
#  ssh connection
#RUN mkdir -p /root/.ssh && \
#    chmod 0700 /root/.ssh

# Copying known hosts to allow my main 
#  pc (client) to ssh into the docker container 
#  just as it would the pc running this container (server)
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fi>
#COPY ./known_hosts  /root/.ssh/known_hosts

# Add the keys and set permissions
#RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
#    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
#    chmod 600 /root/.ssh/id_rsa && \
#    chmod 600 /root/.ssh/id_rsa.pub

# Avoid cache purge by adding requirements first
#ADD ./requirements.txt /code/requirements.txt
# COPY requirements.txt requirements.txt



# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade open3d==0.18.0 && \
    python3 -m pip install --no-cache-dir --upgrade paramiko

WORKDIR /o3d_rtc

#ENV RUN_ENVIRONMENT docker
ENV WEBRTC_IP=0.0.0.0
ENV WEBRTC_PORT=5001


# Install Open3D from the PyPI repositories
#RUN python3 -m pip install --no-cache-dir --upgrade pip && \
 #   python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

# Remove SSH keys for security reasons
#RUN rm -rf /root/.ssh/

# Configure Mesa EGL for headless rendering
ENV EGL_PLATFORM=surfaceless

COPY / /o3d_rtc

ENTRYPOINT ["python3", "rtc_server.py"]
