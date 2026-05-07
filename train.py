import argparse
import datetime
import gorilla
import os
import os.path as osp
import shutil
import time
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

from SMFormer.dataset import build_dataloader, build_dataset
from SMFormer.model import SMFormer_hybrid
from SMFormer.utils import AverageMeter,get_root_logger


#GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 



from SMFormer.dataset import Mangal_ins




def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)
        loss, log_vars = model(batch, mode='loss')

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time and print
        remain_iter = len(dataloader) * (512 - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']
        if i % 10 == 0:
            log_str = f'Epoch [{epoch}/{512}][{i}/{len(dataloader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, '
            log_str += f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)

    # update lr
    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # log and save
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    
    #每10个epoch 保存
    if epoch%10 == 0 or epoch == 1:
        save_file = osp.join("/home/", "SegMangalFormer_"+ str(epoch)+ '.pth')
        meta = dict(epoch=epoch)
        gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)
    
    save_file = osp.join("/home/", 'SegMangalFormer.pth')
    meta = dict(epoch=epoch)
    gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)





if __name__ == "__main__":
    
    log_file = "/home/exptree1123.log"
    logger = get_root_logger(log_file=log_file)

    traindata = Mangal_ins.ForInsDataset("/home/data/", "train","_inst_nostuff.pth",radius =8,logger=logger) #1227


    from config import cfg

    model = SMFormer_hybrid(**cfg)

    #Trainloader
    #sampler = DistributedSampler(traindata, shuffle=True) if dist else None
    # if sampler is not None:
    #     shuffle = False
    train_loader = DataLoader(
            traindata,
            batch_size=10,
            num_workers=5,
            collate_fn=traindata.collate_fn,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True)

    optimizer = gorilla.build_optimizer(model, cfg["optimizer"])
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg["lr_scheduler"])
    #writer
    writer = SummaryWriter("/home/1227.log")
    # pretrain or resume 1021
    start_epoch = 1
    for epoch in range(start_epoch,512 + 1):
        train(epoch, model, train_loader, optimizer, lr_scheduler, cfg, logger, writer)



