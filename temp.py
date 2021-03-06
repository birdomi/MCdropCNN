import os
import argparse
import torch
import timm.utils

from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from adabelief_pytorch import AdaBelief
from utils import get_loader_domainNet

import models

LR = 1e-2
WD = 1e-2
PROPOSAL_NUM = 6
CAT_NUM = 4
save_path = 'checkpoint'

def train():
    
    # get loader
    trainloader, testloader = get_loader_domainNet('D:/DomainNet/', 'real', 'A', 8)

    # define model
    net = models.attention_net(topN=PROPOSAL_NUM)
    # if resume:
    #     ckpt = torch.load(resume)
    #     net.load_state_dict(ckpt['net_state_dict'])
    #     start_epoch = ckpt['epoch'] + 1
    creterion = torch.nn.CrossEntropyLoss()

    # define optimizers
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    raw_optimizer =     AdaBelief(raw_parameters, lr = LR, eps=1e-8, weight_decay=WD, rectify=False, print_change_log = False)
    concat_optimizer =  AdaBelief(concat_parameters, lr = LR, eps=1e-8, weight_decay=WD, rectify=False, print_change_log = False)
    part_optimizer =    AdaBelief(part_parameters, lr = LR, eps=1e-8, weight_decay=WD, rectify=False, print_change_log = False)
    partcls_optimizer = AdaBelief(partcls_parameters, lr = LR, eps=1e-8, weight_decay=WD, rectify=False, print_change_log = False)
    schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
    net = net.cuda()
    saver = timm.utils.CheckpointSaver(net, raw_optimizer, checkpoint_dir= save_path, max_history = 2)

    for epoch in range(0, 500):      
        # begin training
        print('--' * 50)
        net.train()
        for i, data in enumerate(trainloader):
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()

            raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
            part_loss = models.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = models.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()

        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size                    

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))
        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))
        for scheduler in schedulers:
            scheduler.step()
    
        # save model
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        saver.save_checkpoint(epoch, test_acc)

if __name__ == '__main__':
    train()

