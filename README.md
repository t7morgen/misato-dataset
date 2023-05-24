<div align="center">

# MISATO - Machine learning dataset for structure-based drug discovery

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.8+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

</div>
 
## :earth_americas: Where we are:
- Quantum Mechanics: 19443 ligands, curated and refined
- Molecular Dynamics: 16972 simulated protein-ligand structures, 10 ns each 
- AI: pytorch dataloaders, 2 base line models for MD and QM

## :electron: Vision:
We are a drug discovery community project :hugs:
- highest possible accuracy for ligand molecules
- represent the systems dynamics in reasonable timescales
- innovative AI models for drug discovery predictions

Lets crack the **100+ ns** MD, **30000+ protein-ligand structures** and a whole new world of **AI models for drug discovery** together.

[Check out the paper!](MISATO.pdf)

![Alt text](logo.jpg?raw=true "MISATO")

## :purple_heart: Community

Want to get hands-on for drug discovery using AI?


[Join our discord server!](https://discord.gg/tGaut92VYB)


## 📌  Introduction 
 
In this repository, we show how to download and apply the Misato database for AI models. You can access the calculated properties of different protein-ligand structures and use them for training in Pytorch based dataloaders. We provide a small sample of the dataset along with the repo.

You can freely download the **FULL MISATO dataset** from Zenodo using the links below:

- MD (133 GiB)
- QM (0.3 GiB)
- electronic densities (6 GiB)
- MD restart and topology files (55 GiB)
 

```bash
wget -O data/MD/h5_files/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
wget -O data/QM/h5_files/QM.hdf5 https://zenodo.org/record/7711953/files/QM.hdf5
```

**Start with the notebook [src/getting_started.ipynb](src/getting_started.ipynb) to :**

- Understand the structure of our dataset and how to access each molecule's properties.
- Load the PyTorch Dataloaders of each dataset.
- Load the PyTorch lightning Datamodules of each dataset.



## 🚀  Quickstart

We recommend to pull our MISATO image from DockerHub or to create your own image (see [docker/](docker/)).  The images use cuda version 11.8. We recommend to install on your own system a version of CUDA that is a least 11.8 to ensure that the drivers work correctly.

```bash
# clone project
git clone https://github.com/sab148/MiSaTo-dataset.git
cd MiSaTo-dataset
```
For singularity use:
```bash
# get the container image
singularity pull docker://sab148/misato-dataset
singularity shell misato.sif
```

For docker use: 

```bash
sudo docker pull sab148/misato-dataset:latest
bash docker/run_bash_in_container.sh
```

<br>


## Project Structure

```
├── data                   <- Project data
│   ├──MD 
│   │   ├── h5_files           <- storage of dataset
│   │   └── splits             <- train, val, test splits
│   └──QM
│   │   ├── h5_files           <- storage of dataset
│   │   └── splits             <- train, val, test splits
│
├── src                    <- Source code
│   ├── data                    
│   │   ├── components           <- Datasets and transforms
│   │   ├── md_datamodule.py     <- MD Lightning data module
│   │   ├── qm_datamodule.py     <- QM Lightning data module
│   │   │
│   │   └── processing           <- Skripts for preprocessing, inference and conversion
│   │      ├──...    
│   ├── getting_started.ipynb     <- notebook : how to load data and interact with it
│   └── inference.ipynb           <- notebook how to run inference
│
├── docker                    <- Dockerfile and execution script 
└── README.md
```

<br>
<br>

<br>


## Installation using your own conda environment

In case you want to use conda for your own installation please create a new misato environment.

In order to install pytorch geometric we recommend to use pip (within conda) for installation and to follow the official installation instructions:[pytorch-geometric/install](
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

Depending on your CUDA version the instructions vary. We show an example for the CUDA 11.8.

```bash
conda create --name misato python=3
conda activate misato
conda install -c anaconda pandas pip h5py
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache
pip install joblib matplotlib
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pytorch-lightning==1.8.3
pip install torch-geometric
pip install ipykernel==5.5.5 ipywidgets==7.6.3 nglview==2.7.7
conda install -c conda-forge nb_conda_kernels

```

To run inference for MD you have to install ambertools. We recommend to install it in a separate conda environment.

```bash
conda create --name ambertools python=3
conda activate ambertools
conda install -c conda-forge ambertools nb_conda_kernels
pip install h5py jupyter ipykernel==5.5.5 ipywidgets==7.6.3 nglview==2.7.7
```



