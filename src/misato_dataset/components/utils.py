'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''

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