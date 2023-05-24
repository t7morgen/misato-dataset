FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04
USER root
# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime



# Install torch geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install torch_geometric
RUN pip install h5py ipykernel==5.5.5 ipywidgets==7.6.3 jupyter nglview==2.7.7 pandas
RUN pip install pytorch-lightning==1.8.3

RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive; \
    apt-get install -y --no-install-recommends --allow-downgrades\
        bzip2=1.0.8-5build1 cmake=3.22.1-1ubuntu1.22.04.1 csh=20110502-7 \
        make=4.3-4.1build1 gcc=4:11.2.0-1ubuntu1 gfortran=4:11.2.0-1ubuntu1 \
        g++=4:11.2.0-1ubuntu1 flex=2.6.4-8build2 bison=2:3.8.2+dfsg-1build1 \
        patch=2.7.6-7build2 bc=1.07.1-3build1 libbz2-dev=1.0.8-5build1 \
        wget=1.21.2-2ubuntu1 openmpi-bin=4.1.2-2ubuntu1 \
        libopenmpi-dev=4.1.2-2ubuntu1 openssh-client=1:8.9p1-3 \
        ca-certificates=20211016ubuntu0.22.04.1

WORKDIR /usr/bin
COPY AmberTools22.tar.bz2 .
RUN tar xjvf AmberTools22.tar.bz2 && rm AmberTools22.tar.bz2

WORKDIR amber22_src
WORKDIR build
RUN chmod +x run_cmake
RUN ./run_cmake
RUN make -j 4 install
RUN echo "source /usr/bin/amber22/amber.sh" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-c"]

ENV AMBERHOME="/usr/bin/amber22/"
ENV PATH="$AMBERHOME/bin:$PATH"
ENV PYTHONPATH="$AMBERHOME/lib/python3.10/site-packages"

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser .
USER appuser

