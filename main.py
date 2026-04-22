import torch
#
import train_scheduler
import random as pyrandom
import numpy as np
from privacy_analysis import accounting_analysis as aa
from privacy_analysis import handler as ph

import datasets.cifar10 as dms
import utility


if __name__ == '__main__':
    # import os, torch

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # torch.use_deterministic_algorithms(True)

    arg_setup = utility.parse_args()

    # ckpt_path = f"{arg_setup.log_dir}/states_epoch_9.pth"
    # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # # ---- Python / NumPy ----
    # pyrandom.setstate(ckpt["python_rng"])
    # np.random.set_state(ckpt["numpy_rng"])
    # # ---- Torch CPU ----
    # torch.set_rng_state(ckpt["rng_state"])
    # # ---- Torch CUDA (2 GPU) ----
    # torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    
    ''' model and loaders '''
    # all_datasets, all_loader, dataloader_train_aug = dms.get_all(seed = 1, batchsize_train = arg_setup.expected_batchsize) # the batchsize here is of on meaning
    all_datasets, all_loader = dms.get_all(seed = 1, batchsize_train = arg_setup.expected_batchsize)
    model = dms.model(num_of_classes = dms.num_of_classes).to(dms.device)
    model.device = dms.device
    # model = dms_wide.WRN_16_2().to(dms_wide.device)
    # model.device = dms_wide.device
    
    ''' old train loader'''
    # train_loader = ph.privatized_loader_old(all_datasets[0], arg_setup.expected_batchsize)
    train_loader = ph.privatized_loader(all_datasets[0], arg_setup.expected_batchsize)
    # train_loader = all_loader[0]
    
    ''' total image number for batch parameter computation, pub data, from train data'''
    arg_setup.usable_train_data_samples = len(all_datasets[0])
    
    ''' sampling rate for training private data '''
    sampling_rate = arg_setup.expected_batchsize / len(train_loader.dataset) 
    
    ''' compute dp noise '''
    arg_setup.sigma = aa.get_std(q = arg_setup.expected_batchsize / (arg_setup.usable_train_data_samples),
                                                 EPOCH = arg_setup.EPOCH, epsilon = arg_setup.epsilon, delta = 1e-5, verbose = True)
    
    arg_setup.iter_num = int(arg_setup.EPOCH / (arg_setup.expected_batchsize) * 50000)
    
    arg_setup.C = 0.2
    
    arg_setup.C0 = 0.015

    arg_setup.self_augment = False
    
    # arg_setup.dataloader_train_aug = dataloader_train_aug
    
    '''sgd opti'''
    opti = torch.optim.SGD(model.parameters(), lr = arg_setup.lr, momentum = 0.0 ); arg_setup.beta = 0.9

    '''adam opti'''
    # opti = torch.optim.Adam(model.parameters(), lr = 0.01); arg_setup.beta = 0
    
    ''' function signature '''
    TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')
    train_setups = {
                    'epoch': arg_setup.EPOCH, 
                    'device': dms.device, 
                    'optimizer': opti,
                    'loss_metric': dms.loss_metric, 
                    'enable_per_grad': (True, 'opacus'),
                    'sigma': arg_setup.sigma,
                    }

    trainer = train_scheduler.train_master(
                                            model = model,
                                            loaders = [train_loader, None, all_loader[2]],
                                            train_setups = train_setups,
                                            arg_setup = arg_setup,
                                            
                                            )   
    trainer.train()
