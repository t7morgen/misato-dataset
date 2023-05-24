import os 
import re
import errno

import torch
import torch.distributed as dist

def _get_first_node():
    """Return the first node we can find in the Slurm node list."""
    nodelist = os.getenv('SLURM_JOB_NODELIST')

    bracket_re = re.compile(r'(.*?)\[(.*?)\]')
    dash_re = re.compile('(.*?)-')
    comma_re = re.compile('(.*?),')

    bracket_result = bracket_re.match(nodelist)

    if bracket_result:
        node = bracket_result[1]
        indices = bracket_result[2]

        comma_result = comma_re.match(indices)
        if comma_result:
            indices = comma_result[1]

        dash_result = dash_re.match(indices)
        if dash_result:
            first_index = dash_result[1]
        else:
            first_index = indices

        return node + first_index

    comma_result = comma_re.match(nodelist)
    if comma_result:
        return comma_result[1]

    return nodelist


def init_distributed_mode(port=12354):
    """Initialize some environment variables for PyTorch Distributed
    using Slurm.
    """
    # The number of total processes started by Slurm.
    os.environ['WORLD_SIZE'] = os.getenv('SLURM_NTASKS')
    # Index of the current process.
    os.environ['RANK'] = os.getenv('SLURM_PROCID')
    # Index of the current process on this node only.
    os.environ['LOCAL_RANK'] = os.getenv('SLURM_LOCALID')

    master_addr = _get_first_node()
    systemname = os.getenv('SYSTEMNAME', '')
    # Need to append "i" on Jülich machines to connect across InfiniBand cells.
    if systemname in ['juwels', 'juwelsbooster', 'jureca']:
        master_addr = master_addr + 'i'
    os.environ['MASTER_ADDR'] = master_addr

    # An arbitrary free port on node 0.
    os.environ['MASTER_PORT'] = str(port)
    return os.environ['WORLD_SIZE'], os.environ['RANK'], os.environ['LOCAL_RANK']

# Returns True if the distributed package is available. False otherwise
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True 

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise