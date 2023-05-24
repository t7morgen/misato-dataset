import os
import subprocess

def download_data(self, name, out_path):
    """ Download data if needed.
        Available datasets are MD, QM. 

    Args:
        name (str): Two-letter code for dataset (not case-sensitive).
        out_path (str): Path to directory in which to save downloaded dataset.
    """
    if name == "qm":
        link = 'https://zenodo.org/record/4911102/files/PPI-raw.tar.gz?download=1'
    elif name == "md":
        link = 'https://zenodo.org/record/4911102/files/PPI-raw.tar.gz?download=1'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cmd = f"wget {link} -O {out_path}/{name}.tar.gz"
    subprocess.call(cmd, shell=True)
    cmd2 = f"tar xzvf {out_path}/{name}.tar.gz -C {out_path}"
    subprocess.call(cmd2, shell=True)