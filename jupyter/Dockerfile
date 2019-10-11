FROM python:3.6

WORKDIR /home/jupyterlab

COPY requirements.txt /home/jupyterlab

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . /home/jupyterlab
RUN jupyter nbextension enable --sys-prefix --py widgetsnbextension
RUN jupyter serverextension enable --py nbresuse --sys-prefix
RUN jupyter nbextension install --py nbresuse --sys-prefix
RUN jupyter nbextension enable --py nbresuse --sys-prefix

CMD jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''