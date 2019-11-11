FROM python:3.6

WORKDIR /home/jupyterlab

# The java version javabridge (> python-weka-wrapper3 > weka) needs (8) is removed
# from the debian package repository. These lines tell apt to use the back door:
RUN echo 'deb http://deb.debian.org/debian sid main' >> /etc/apt/sources.list

RUN apt-get update
RUN apt-get -y install openjdk-8-jdk

# turn off unstable packages again
RUN head -n -1 /etc/apt/sources.list > temp.txt ; mv temp.txt /etc/apt/sources.list

# required for exporting notebooks to other formats
RUN apt-get -y install pandoc

COPY requirements.txt /home/jupyterlab

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . /home/jupyterlab
RUN jupyter nbextension enable --sys-prefix --py widgetsnbextension
RUN jupyter serverextension enable --py nbresuse --sys-prefix
RUN jupyter nbextension install --py nbresuse --sys-prefix
RUN jupyter nbextension enable --py nbresuse --sys-prefix

CMD jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''