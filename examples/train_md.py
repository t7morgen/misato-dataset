import sys
import argparse
import logging
import os
import time
import datetime

#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from scipy.stats import spearmanr

from MDmodel import GNN_MD
import utils
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)).split('MiSaTo-dataset')[0],'MiSaTo-dataset/src/data/components/'))

from datasets import ProtDataset
from transformMD import GNNTransformMD

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

    total_loss = torch.tensor([np.sqrt(loss_all / total)]).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)

    return total_loss


@torch.no_grad()
def test(model, loader, local_rank):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(f'cuda:{local_rank}')
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())

    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]


    total_loss = torch.tensor([np.sqrt(loss_all / total)]).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)
    
    r_p = torch.tensor([np.sqrt(np.corrcoef(y_true, y_pred)[0,1])]).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(r_p, op=torch.distributed.ReduceOp.AVG)
    
    r_s = torch.tensor([spearmanr(y_true, y_pred)[0]]).to(f'cuda:{local_rank}')
    torch.distributed.all_reduce(r_s, op=torch.distributed.ReduceOp.AVG)
    
    return total_loss, r_p, r_s, y_true, y_pred

# def plot_corr(y_true, y_pred, plot_dir):
#     plt.clf()
#     sns.scatterplot(y_true, y_pred)
#     plt.xlabel('Actual -log(K)')
#     plt.ylabel('Predicted -log(K)')
#     plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, local_rank, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    train_dataset = ProtDataset(args.mdh5_file, idx_file=args.train_set, transform=GNNTransformMD(), post_transform=T.RandomTranslate(0.05))
    val_dataset = ProtDataset(args.mdh5_file, idx_file=args.val_set, transform=GNNTransformMD(), post_transform=T.RandomTranslate(0.05))
    test_dataset = ProtDataset(args.mdh5_file, idx_file=args.test_set, transform=GNNTransformMD(), post_transform=T.RandomTranslate(0.05))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=24, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=24, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=24, sampler=test_sampler)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_MD(num_features, hidden_dim=args.hidden_dim)
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    model = model.to(f'cuda:{local_rank}')
    
    # Synchronization of batchnorm statistics
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # implements data parallelism of the model which can run across multiple machines. 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(local_rank)])
    model_without_ddp = model.module

    best_val_loss = 999
    best_rp = 0
    best_rs = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        train_sampler.set_epoch(epoch)
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, local_rank)
        val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, local_rank)
        val_loss = val_loss.item()
        if utils.is_main_process():
            if val_loss < best_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss.item(),
                    }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
                best_val_loss = val_loss
                best_rp = r_p.item()
                best_rs = r_s.item()
        elapsed = (time.time() - start)
        if utils.is_main_process():
            print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
            print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss.item(), val_loss, r_p.item(), r_s.item()))
            # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

    if test_mode:
        train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        if utils.is_main_process():
            model.load_state_dict(cpt['model_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(model, train_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(model, val_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        rmse, pearson, spearman, y_true_test, y_pred_test = test(model, test_loader, local_rank)
        if utils.is_main_process():
            print(f'\tTest RMSE {rmse}, Test Pearson {pearson}, Test Spearman {spearman}')
            torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)

    return best_val_loss, best_rp, best_rs


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdh5_file', type=str, default="../data/MD/h5_files/MD_out.hdf5")
    parser.add_argument('--train_set', type=str, default="../data/MD/splits/train_MD.txt")
    parser.add_argument('--val_set', type=str, default="../data/MD/splits/val_MD.txt")
    parser.add_argument('--test_set', type=str, default="../data/MD/splits/test_MD.txt")
    parser.add_argument( '--master_port', type=int, default=12354)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
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
            log_dir = os.path.join('logs', f'MD_{now}')
            print("log_dir", log_dir)
            if utils.is_main_process():
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, local_rank, rep, test_mode=True)