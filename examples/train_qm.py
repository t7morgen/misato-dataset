import sys
import argparse
import logging
import os
import time
import datetime

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


from QMmodel import GNN_QM
import utils
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)).split('MiSaTo-dataset')[0],'MiSaTo-dataset/src/data/components/'))

from datasets import MolDataset
from transformQM import GNNTransformQM



def train_loop(model, loader, optimizer, local_rank):
    model.train()

    loss_all = 0
    total = 0
    for data in loader:
        data = data.to(f'cuda:{local_rank}')
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()


    total_loss = torch.tensor([loss_all / total]).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)

    return total_loss

@torch.no_grad()
def test(model, loader, local_rank):
    model.eval()
    loss_all = 0
    total = 0
    y_true = []
    y_pred = []
    y_id = []
    for data in loader:
        data = data.to(f'cuda:{local_rank}')
        output=model(data)
        loss = F.l1_loss(output, data.y)  # MAE
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend([x.item() for x in data.y])
        y_pred.extend(output.tolist())
        y_id.extend(data.id)

    total_loss = torch.tensor(loss_all / total).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)

    return total_loss, y_true, y_pred, y_id


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, local_rank, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    isTrainval = True
    print("isTrain", isTrainval)
    train_dataset = MolDataset(args.qmh5_file, args.train_set, target_norm_file=args.norm_file, transform=GNNTransformQM(), isTrain=isTrainval, post_transform=T.RandomTranslate(0.25))
    val_dataset = MolDataset(args.qmh5_file, args.val_set, target_norm_file=args.norm_file, transform=GNNTransformQM(), post_transform=T.RandomTranslate(0.25))
    test_dataset = MolDataset(args.qmh5_file, args.test_set, target_norm_file=args.norm_file, transform=GNNTransformQM(), post_transform=T.RandomTranslate(0.25))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
           
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=24, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=24, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=24, sampler=test_sampler)

    for data in train_loader:
        num_features = data.num_features
        break


    model = GNN_QM(num_features, dim=args.hidden_dim).to(device)
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    model = model.to(f'cuda:{local_rank}')
    
    # Synchronization of batchnorm statistics
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # implements data parallelism of the model which can run across multiple machines. 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(local_rank)])
    model_without_ddp = model.module

    best_val_loss = 999


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=0.00001)

    for epoch in range(1, args.num_epochs+1):
        train_sampler.set_epoch(epoch)
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, local_rank)
        if utils.is_main_process():
            print('validating...')
        val_loss,  _,_, _ = test(model, val_loader, local_rank)
        val_loss = val_loss.item()
        scheduler.step(val_loss)
        if utils.is_main_process():
            if val_loss < best_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss.item(),
                    }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                best_val_loss = val_loss
        elapsed = (time.time() - start)
        if utils.is_main_process():
            print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
            print('\tTrain Loss: {:.7f}, Val MAE: {:.7f}'.format(train_loss.item(), val_loss))

    if test_mode:
        train_file = os.path.join(log_dir, f'smp-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'smp-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'smp-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model_without_ddp.load_state_dict(cpt['model_state_dict'])
        _, y_true_train, y_pred_train, y_id = test(model, train_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_train, 'predictions':y_pred_train, 'ids':y_id}, train_file)
        _, y_true_val, y_pred_val, y_id = test(model, val_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_val, 'predictions':y_pred_val, 'ids':y_id}, val_file)
        mae, y_true_test, y_pred_test, y_id = test(model, test_loader, local_rank)
        if utils.is_main_process():
            print(f'\tTest MAE {mae}')
            torch.save({'targets':y_true_test, 'predictions':y_pred_test, 'ids':y_id}, test_file)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qmh5_file', type=str, default="/p/project/atmlaml/benassou1/misato/MiSaTo-dataset/data/QM/h5_files/qm.hdf5")
    parser.add_argument('--norm_file', type=str, default="/p/project/atmlaml/benassou1/misato/MiSaTo-dataset/data/QM/h5_files/qm_norm_fold.hdf5")
    parser.add_argument('--train_set', type=str, default="/p/project/atmlaml/benassou1/misato/MiSaTo-dataset/data/QM/splits/train_norm.txt")
    parser.add_argument('--val_set', type=str, default="/p/project/atmlaml/benassou1/misato/MiSaTo-dataset/data/QM/splits/val_norm.txt")
    parser.add_argument('--test_set', type=str, default="/p/project/atmlaml/benassou1/misato/MiSaTo-dataset/data/QM/splits/test_norm.txt")
    parser.add_argument( '--master_port', type=int, default=12354)
    parser.add_argument('--target_name', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    # set up the environement variable for the distributed mode
    world_size, rank, local_rank = utils.init_distributed_mode(args.master_port)

    # Initialize the torch.distributed package
    torch.distributed.init_process_group(backend='nccl')

    # return an object representing the device on which tensors will be allocated.
    device = torch.device(device)

    # enable benchmark mode in cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if utils.is_main_process():
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        train(args, device, log_dir, local_rank)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', f'QM_{now}')
            
            if utils.is_main_process():
                print("log_dir", log_dir)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, local_rank, rep, test_mode=True)