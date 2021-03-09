from __future__ import print_function

import argparse
import os
import sys
import warnings
import shutil
import time
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
import torch.utils.data as data
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from models.imagenet.resnet import ResNetDCT_Upscaled_Static
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets.dataloader_imagenet_dct import valloader_upscaled_static, trainloader_upscaled_static, test_transform,testloader_upscaled_static
from tensorboardX import SummaryWriter

import cv2
import csv

from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from datasets import train_y_mean, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets import train_dct_subset_mean, train_dct_subset_std
from datasets import train_upscaled_static_mean, train_upscaled_static_std
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch',
                    default=32,
                    type=int,
                    metavar='N',
                    help='test batchsize (default: 32)')
parser.add_argument('--test-batch',
                    default=32,
                    type=int,
                    metavar='N',
                    help='test batchsize (default: 32)')
parser.add_argument('-c',
                    '--checkpoint',
                    default='checkpoints',
                    type=str,
                    metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50dct',
                    help='model architecture: (default: resnet50dct)')
# Miscs
parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='gloo',
                    type=str,
                    help='distributed backend')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--subset',
                    default='192',
                    type=str,
                    help='subset of y, cb, cr')
#Device options
parser.add_argument('--gpu-id',
                    default='0',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained',
                    default='False',
                    type=str2bool,
                    help='load pretrained model or not')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_prec1 = 0  # best test accuracy


def main():
    global args, best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size)
    # print(args.resume)

    # model = torch.load(args.resume)

    model = ResNetDCT_Upscaled_Static(channels=int(args.subset),
                                      input_gate=True,
                                      pretrained=False)
    model.fc1 = nn.Linear(model.fc1.in_features, 49)
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    print(args.resume)
    model.load_state_dict(torch.load(args.resume))
    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=0.001,
    #                             momentum=0.9,
    #                             weight_decay=5e-4)

    # # Resume
    # print("1")
    # title = 'ImageNet-' + args.arch
    # if not os.path.isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load(args.resume)
    #     best_prec1 = checkpoint['best_prec1']
    #     model_dict = model.state_dict()
    #     pretrained_dict = {
    #         k: v
    #         for k, v in checkpoint['state_dict'].items() if k in model_dict
    #     }
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict=False)
    #     #model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     args.checkpoint = os.path.dirname(args.resume)

    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model = model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    # cudnn.benchmark = True
    # print('Total params: %.2fM' %
    #       (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=10,
    #                                             gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [160, 180],
    #                                                  0.1)
    test_loader, test_val = testloader_upscaled_static(args, model='resnet')
    print('Evaluation only')
    test_model(
        model,
        test_loader,
    )
    # test1(model, val_loader)
    # test(model)
    return


def test_model(model, dataloader):
    model.eval()
    csvfile = open('./csv.csv', 'w')
    writer = csv.writer(csvfile)
    # predictions = np.zeros(size)
    # all_classes = np.zeros(size)
    # all_proba = np.zeros((size, 2))
    i=0
    with torch.no_grad():
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            # print(classes)
            # print(outputs)
            # loss = CB_loss(labels=classes,
            #                logits=outputs,
            #                samples_per_cls=cls_num,
            #                no_of_classes=49,
            #                loss_type="sigmoid",
            #                beta=0.9999,
            #                gamma=2)
            _, preds = torch.max(outputs.data, 1)
            for j in preds.tolist():
                print(str(i) + ":" + str(j))
                writer.writerow([i, j])
                i += 1
            # print(preds)
            # print(classes.data)
            # print('*****************')
            # statistics
            # running_loss += loss.data.item()
            # running_corrects += torch.sum(preds == classes.data)
            #predictions[i:i+len(classes)] = preds.to('cpu').numpy()
            #all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
            #all_proba[i:i+len(classes),:] = outputs.data.to('cpu').numpy()
            # i += len(classes)
            #print('Testing: No. ', i, ' process ... total: ', size)
    # epoch_loss = running_loss / size
    # epoch_acc = running_corrects.data.item() / size
    #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return


def test1(model, dataloader):
    csvfile = open('./csv.csv', 'w')
    writer = csv.writer(csvfile)
    model.eval()
    # predictions = np.zeros(size)
    # all_classes = np.zeros(size)
    # all_proba = np.zeros((size, 2))
    i = 0
    with torch.no_grad():
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)

            # print(classes)
            # print(outputs)
            # loss = CB_loss(labels=classes,
            #                logits=outputs,
            #                samples_per_cls=cls_num,
            #                no_of_classes=49,
            #                loss_type="sigmoid",
            #                beta=0.9999,
            #                gamma=2)
            _, preds = torch.max(outputs.data, 1)
            # statistics
            print(i, preds.tolist())
            # for j in preds.tolist():
            #     print(str(i)+":"+str(j))
            #     writer.writerow([i, j])
            #     i+=1
            #predictions[i:i+len(classes)] = preds.to('cpu').numpy()
            #all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
            #all_proba[i:i+len(classes),:] = outputs.data.to('cpu').numpy()

            #print('Testing: No. ', i, ' process ... total: ', size)

    #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return


def test(model):
    # bar = Bar('Processing', max=len(val_loader))

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    csvfile = open('./csv.csv', 'w')
    writer = csv.writer(csvfile)
    test_root = './data/test/'
    img_test = os.listdir(test_root)
    img_test.sort(key=lambda x: int(x[:-4]))

    input_size1 = 512
    input_size2 = 448

    transform = transforms.Compose([
        transforms.Resize(input_size1),
        transforms.CenterCrop(input_size2),
        transforms.Upscale(upscale_factor=2),
        transforms.TransformUpscaledDCT(),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(channels=args.subset),
        transforms.Aggregate(),
        transforms.NormalizeDCT(train_upscaled_static_mean,
                                train_upscaled_static_std,
                                channels=args.subset)
    ])

    with torch.no_grad():
        # end = time.time()
        for i in range(len(img_test)):
            model.eval()
            # measure data loading time
            # data_time.update(time.time() - end)

            # image, target = image.cuda(non_blocking=True), target.cuda(
            #     non_blocking=True)

            image = cv2.imread(str(test_root + img_test[i]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(transform(image)[0])
            # print(type(transform(image)[0]))
            # compute output
            output = model(transform(image)[0].unsqueeze(dim=0))
            #print(output)
            _, pred = torch.max(output.data, 1)
            print(i, pred.tolist()[0])
            writer.writerow([i, pred.tolist()[0]])
            # loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            # losses.update(loss.item(), image.size(0))
            # top1.update(prec1.item(), image.size(0))
            # top5.update(prec5.item(), image.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # plot progress
            # bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            #     batch=batch_idx + 1,
            #     size=len(val_loader),
            #     data=data_time.avg,
            #     bt=batch_time.avg,
            #     total=bar.elapsed_td,
            #     eta=bar.eta_td,
            #     loss=losses.avg,
            #     top1=top1.avg,
            #     top5=top5.avg,
            # )
            # bar.next()
        # bar.finish()


if __name__ == '__main__':
    main()