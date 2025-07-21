FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

# --------------------------------------------- #
# Preconfigure the timezone
# --------------------------------------------- #
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# --------------------------------------------- #
# Install dependencies
# --------------------------------------------- #
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop \
    libgl1-mesa-glx libglib2.0-0 gnupg g++ build-essential python3-tk \
    v4l-utils software-properties-common screen openssh-client ffmpeg \
    unzip unrar \
    && apt clean 

# --------------------------------------------- #
# Install Python
# --------------------------------------------- #
ARG PYTHON_VERSION
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update \
    && apt install --no-install-recommends -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    && apt clean

# --------------------------------------------- #
# Setup Python virtual environment
# --------------------------------------------- #
ARG PYTHON_VENV_DIR
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir --upgrade pip
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir virtualenv
RUN python${PYTHON_VERSION} -m virtualenv \
    ${PYTHON_VENV_DIR} --python=python${PYTHON_VERSION}

# # --------------------------------------------- #
# # Install Python packages
# # --------------------------------------------- #
# Install Python packages
COPY requirements.txt /tmp/requirements.txt

RUN source ${PYTHON_VENV_DIR}/bin/activate && \
    python -m pip install -r /tmp/requirements.txt


# --------------------------------------------- #
# Setup working directory
# --------------------------------------------- #
ARG WORKDIR
WORKDIR ${WORKDIR}

# --------------------------------------------- #
# Setup user
# --------------------------------------------- #
ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} ${USER_NAME} \
    && useradd -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME} \
    && install -d -m 0754 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME} \
    && chown --silent --no-dereference --recursive \
    ${USER_ID}:${GROUP_ID} \
    /home/${USER_NAME} \
    ;fi

#RUN chown --recursive --silent ${USER_NAME}:${USER_NAME} ${WORKDIR} \
    #&& chown --recursive --silent ${USER_NAME}:${USER_NAME} ${PYTHON_VENV_DIR}

# --------------------------------------------- #
# Setup vision group
# --------------------------------------------- #
# ARG VISION_GROUP_ID
# RUN if [ ${VISION_GROUP_ID:-0} -ne 0 ] \ 
#     && ! getent group | cut -d: -f3 | grep -q "^${VISION_GROUP_ID}$"; then \
#     groupadd -g ${VISION_GROUP_ID} vision \
#     ;fi
# RUN usermod -aG ${VISION_GROUP_ID} ${USER_NAME}

RUN usermod -aG audio ${USER_NAME}
USER ${USER_NAME}

# --------------------------------------------- #
# Setup virtual environment as default
# --------------------------------------------- #
RUN echo "source ${PYTHON_VENV_DIR}/bin/activate" >> ~/.bashrc
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]