FROM nvcr.io/nvidia/pytorch:23.02-py3

# Install pip requirements
COPY requirements.txt .

RUN apt-get update && env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip python3-venv python-tk
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir jupyter

RUN pip3 install --no-cache-dir -r requirements.txt

RUN env CPATH="/usr/local/cuda/include:$CPATH" FORCE_CUDA=1 pip3 install --no-cache-dir torch-scatter torch-sparse
RUN pip3 install --no-cache-dir torch-geometric

RUN env DEBIAN_FRONTEND=noninteractive apt-get install -y tcsh make \
        gcc gfortran \
        flex bison patch bc wget \
        xorg-dev libz-dev libbz2-dev

WORKDIR /usr/bin
COPY AmberTools22.tar.bz2 .
RUN tar xjf AmberTools22.tar.bz2 && rm AmberTools22.tar.bz2

WORKDIR amber22_src/build
RUN chmod +x run_cmake
RUN sed -i 's/-DDOWNLOAD_MINICONDA=TRUE/-DDOWNLOAD_MINICONDA=FALSE/g' run_cmake
RUN env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk
RUN ./run_cmake
RUN make -j4 install
RUN echo "source /usr/bin/amber22/amber.sh" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-c"]

ENV AMBERHOME="/usr/bin/amber22"
ENV PATH="$AMBERHOME/bin:$PATH"
ENV PYTHONPATH="$AMBERHOME/lib/python3.8/site-packages:$PYTHONPATH"

COPY ULYSSES.py .

RUN python ULYSSES.py

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser .
USER appuser

