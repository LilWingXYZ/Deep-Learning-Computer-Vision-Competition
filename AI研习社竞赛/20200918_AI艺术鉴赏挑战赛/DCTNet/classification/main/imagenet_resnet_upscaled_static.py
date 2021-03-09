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
from datasets.dataloader_imagenet_dct import valloader_upscaled_static, trainloader_upscaled_static
from tensorboardX import SummaryWriter
from class_balanced_loss import ClassBalancedLoss


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

    model = ResNetDCT_Upscaled_Static(channels=int(args.subset),
                                      pretrained=args.pretrained,
                                      input_gate=True)
    model.fc1 = nn.Linear(model.fc1.in_features, 49)
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=5e-4)

    # Resume
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in checkpoint['state_dict'].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(
            args.resume, checkpoint['epoch']))
        args.checkpoint = os.path.dirname(args.resume)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code
    train_loader, len_train, cls_num = trainloader_upscaled_static(
        args, model='resnet')

    val_loader, len_val = valloader_upscaled_static(args, model='resnet')

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=10,
    #                                             gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [160, 180],
                                                     0.1)

    print(model)
    print(cls_num)
    criterion = ClassBalancedLoss(samples_per_class=cls_num, loss_type="focal").cuda()
    train_model(model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_size=len_train,
                val_size=len_val,
                epochs=500,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                cls_num=cls_num)

    # if args.evaluate:
    #     print('\nEvaluation only')
    #     test_loss, test_acc_top1, test_acc_top5 = test(val_loader, model,
    #                                                    criterion)
    #     print(' Test Loss:  %.8f, Test Acc Top1:  %.2f, Test Acc Top5:  %.2f' %
    #           (test_loss, test_acc_top1, test_acc_top5))
    #     return


def test(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (image, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            image, target = image.cuda(non_blocking=True), target.cuda(
                non_blocking=True)

            # compute output
            output = model(image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def val_model(model, dataloader, size, criterion, cls_num):
    model.eval()
    # predictions = np.zeros(size)
    # all_classes = np.zeros(size)
    # all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.cuda(), classes.cuda()).cuda()
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
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
            #predictions[i:i+len(classes)] = preds.to('cpu').numpy()
            #all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
            #all_proba[i:i+len(classes),:] = outputs.data.to('cpu').numpy()
            i += len(classes)
            #print('Testing: No. ', i, ' process ... total: ', size)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def train_model(model,
                train_loader,
                val_loader,
                train_size,
                val_size,
                scheduler,
                criterion,
                cls_num,
                epochs=1,
                optimizer=None):
    min_loss = 100000
    max_acc = 0
    path = './weights'
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0
        count = 0
        for inputs, classes in train_loader:
            inputs = inputs.to(device)
            classes = classes.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.cuda(), classes.cuda()).cuda()
            # print(classes)
            # print(outputs)
            # print(len(outputs[0]))
            # loss = CB_loss(labels=classes,
            #                logits=outputs,
            #                samples_per_cls=cls_num,
            #                no_of_classes=49,
            #                loss_type="sigmoid",
            #                beta=0.9999,
            #                gamma=2)
            optimizer = optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
            # print(preds)
            # print(classes.data)
            # print('*****************')
            count += len(inputs)
            # print('Training: No. ', count, ' process ... total: ', size)
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.data.item() / train_size
        epoch_Valloss, epoch_Valacc = val_model(model,
                                                val_loader,
                                                val_size,
                                                criterion=criterion,
                                                cls_num=cls_num)
        #print("current lr: "+model.optimizer.state_dict()['param_groups'][0]['lr'])
        print(
            'epoch: ', epoch,
            ' Loss: {:.5f} Acc: {:.5f} ValLoss: {:.5f} ValAcc: {:.5f}'.format(
                epoch_loss, epoch_acc, epoch_Valloss, epoch_Valacc))
        if epoch_Valloss < min_loss:
            torch.save(model, path + '/minloss_model.pth')
            print("save min loss model")
            min_loss = epoch_Valloss
        elif epoch_Valacc > max_acc:
            torch.save(model, path + '/maxacc_model.pth')
            print("save max acc model")
            max_acc = epoch_Valacc

        scheduler.step()


if __name__ == '__main__':
    main()
