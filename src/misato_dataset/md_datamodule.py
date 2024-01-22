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

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from .components.datasets import ProtDataset
from .components.transformMD import GNNTransformMD

class MDDataModule(LightningDataModule):
    """A DataModule implements 4 key methods:

        def setup(self, stage):
            # things to do on every process
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
    """

    def __init__(
        self,
        files_root: str,
        h5file = "h5_files/MD_dataset_soft_hard_noH.hdf5",
        train = "splits/train_soft_hard.txt",
        val = "splits/val_soft_hard.txt",
        test = "splits/test_soft_hard.txt",
        batch_size = 16,
        num_workers = 48,
        transform = T.RandomTranslate(0.05)
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)


        self.files_root = files_root
        self.h5file = h5file

        self.train = train
        self.val = val
        self.test = test

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        

    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """

        self.data_train = ProtDataset(os.path.join(self.files_root, self.h5file), os.path.join(self.files_root, self.train), transform=GNNTransformMD(), post_transform=self.transform)
        self.data_val = ProtDataset(os.path.join(self.files_root, self.h5file), os.path.join(self.files_root, self.val), transform=GNNTransformMD())
        self.data_test = ProtDataset(os.path.join(self.files_root, self.h5file), os.path.join(self.files_root, self.test), transform=GNNTransformMD())

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    _ = MDDataModule()
