FROM deepbase/pytorch:1.8.1

ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker

COPY . /tmp
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && cp /root/.zshrc /home/${USER_NAME}/ ; fi

USER ${USER_NAME}
