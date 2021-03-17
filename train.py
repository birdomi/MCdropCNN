import torch
import timm
import os
import timm.utils

from utils import *
import models

import argparse
from adabelief_pytorch import AdaBelief

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--model_name', default='resnet34', type=str)
    parser.add_argument('--dataset', default='D:/DomainNet/', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_path', default='dropResNet/', type=str)
    args = parser.parse_args()


    main_cuda = 'cuda:'+args.gpu

    print(args.model_name, args.dataset.split('/')[-2], args.batch_size)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    else:
        file_list = os.listdir(args.save_path)


    # Load student model
    model = models.resnet34(
        num_classes = 173,
        dropout=True
    )
    #if len(devices)>1:
    model.to(main_cuda)   

    #print(teacher_model(torch.rand(1,3,224,224).to(main_cuda)).shape)

    train_loader, valid_loader = get_loader_domainNet(args.dataset, 'real', 'A', args.batch_size)

    CEL = torch.nn.CrossEntropyLoss()
    optimizer = AdaBelief(model.parameters(), lr = 1e-3, eps=1e-8, weight_decay=1e-2, rectify=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= args.save_path, max_history = 2)
    
    for epoch in range(90):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(main_cuda), targets.to(main_cuda)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CEL(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs.max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = validation_accuracy(model, valid_loader, main_cuda)
        scheduler.step()
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

if __name__ == '__main__':
    train()