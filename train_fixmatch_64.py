import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from copy import deepcopy
import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.filters import threshold_otsu
from dataset.cifar10_6_4 import DATASET_GETTERS
from sklearn.metrics import f1_score
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    # 返回的学习率调度器是LambdaLR类的实例，这个调度器根据_lr_lambda函数计算当前步骤的学习率
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def f1_score_1(output, target, numclass=100):
    _, maxk = output.max(dim=1,keepdim=False)
    label_one_hot = nn.functional.one_hot(target, numclass).float()
    pred_one_hot = nn.functional.one_hot(maxk, numclass).float()
    tp = label_one_hot * pred_one_hot
    tp = tp.sum(dim=0)
    fn = label_one_hot * (1 - pred_one_hot)
    fn = fn.sum(dim=0)
    fp = (1 - label_one_hot) * pred_one_hot
    fp = fp.sum(dim=0)
    tn = (1 - label_one_hot) * (1 - pred_one_hot)
    tn = tn.sum(dim=0)
    return tp, fp, fn, tn

def selected_eval(args, unlabeled_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    pseudo_list = torch.zeros(args.num_classes, 3)
    pseudo_unmask_num = 0
    end = time.time()
    len_ul = len(unlabeled_loader.dataset)
    if not args.no_progress:
        unlabeled_loader = tqdm(unlabeled_loader)

    with torch.no_grad():
        for batch_idx, ((inputs_w, inputs_s), targets, indexs) in enumerate(unlabeled_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs_w.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            outputs = torch.softmax(outputs.detach()/1, dim=-1)
            prob, pseudo_label = torch.max(outputs, 1)


            mask = prob > 0.95
            pseudo_mask = pseudo_label[mask]
            targets_mask = targets[mask]

            pseudo_unmask_num += (~mask).sum()

            cls_id_tmp = (targets_mask <= 5)
            cls_ood_tmp = ~cls_id_tmp

            targets_mask_id = targets_mask[cls_id_tmp]
            targets_mask_ood = targets_mask[cls_ood_tmp]
            pseudo_mask_id = pseudo_mask[cls_id_tmp]
            pseudo_mask_ood = pseudo_mask[cls_ood_tmp]

            for i in range(args.num_classes):
                idx_tmp_id = (targets_mask_id == i)
                pseudo_true = (pseudo_mask_id[idx_tmp_id] == targets_mask_id[idx_tmp_id]).sum()
                pseudo_false = (pseudo_mask_id == i).sum() - pseudo_true
                ood_false_num = (pseudo_mask_ood == i).sum()

                pseudo_list[i, 0] += pseudo_true.cpu()
                pseudo_list[i, 1] += pseudo_false.cpu()
                pseudo_list[i, 2] += ood_false_num.cpu()
    print(len_ul)
    print(pseudo_list.sum())
    assert len_ul == (pseudo_unmask_num + pseudo_list.sum())
    pseudo_list_np = pseudo_list.numpy()
    print(pseudo_list_np)

    return pseudo_list

def train(args, labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, model, ema_model):

    global best_acc, best_acc_val
    val_accs = []
    test_accs = []
    end = time.time()

    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # 学习率调度器，用于在训练过程中动态调整学习率
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    # 迭代器
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    # 存储在训练过程中计算的召回率、精确度、伪标签信息的多维数组
    recall_np = torch.zeros((args.epochs, 6),device = args.device)
    precision_np = torch.zeros((args.epochs, 6),device = args.device)
    pseudo_list_np = torch.zeros((args.epochs, 6, 3),device = args.device)


    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        model.train()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, index_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, index_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), gt_u, index_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), gt_u, index_u = unlabeled_iter.next()

            data_time.update(time.time() - end)

            batch_size = inputs_x.shape[0]  # size of labeled example
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)   # pb
            # weak augmentation and strong augmentation
            logits = model(inputs)  
            # pm(y|a(xb))
            logits_x = logits[:batch_size]
            # pm(y|a(ub)), pm(y|A(xb))
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            # del logits

            # Cross Entropy Loss for Labeled Data
            # the standard cross-entropy loss on weakly augmented labeled examples
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # Consistency Constraint Loss for Unlabeled Data
            # hyper parameters for UDA
            # T = 0.4

            pseudo_label = torch.softmax(logits_u_w.detach()/1, dim=-1) # qb
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # targets_u : qb^
            mask = max_probs.ge(args.threshold).float()     # ge: greater than or equal
            # unsupervised loss
            Lu = (F.cross_entropy(logits_u_s,
                             targets_u,
                             reduction='none') * mask).mean()

            loss = Lx + Lu

            optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                                      "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        val_acc, val_recall, val_precision = test(args, val_loader, test_model, epoch)
        test_acc, test_recall, test_precision = test(args, test_loader, test_model, epoch)
        pseudo_list = selected_eval(args, unlabeled_trainloader, model, epoch)


        print("\n| Recall:\n", test_recall)
        print("\n| Precision:\n", test_precision)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('val/1.val_acc', val_acc, epoch)


        best_acc_val = max(val_acc, best_acc_val)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        val_accs.append(val_acc)

        recall_np[epoch] = test_recall
        precision_np[epoch] = test_precision
        pseudo_list_np[epoch] = pseudo_list

        recall_np_t = recall_np.cpu().numpy()
        precision_np_t = precision_np.cpu().numpy()
        pseudo_list_np_t = pseudo_list_np.cpu().numpy()

        np.save(args.out + "/test_recall_np.npy", recall_np_t)
        np.save(args.out + "/test_precision_np.npy", precision_np_t)
        np.save(args.out + "/selected_label.npy", pseudo_list_np_t)

        logger.info('save test recall & percision successfully!')
        logger.info('Best top-1 acc(test): {:.2f} | acc(val): {:.2f}'.format(best_acc, best_acc_val))
        logger.info('Mean top-1 acc(test): {:.2f} | acc(val): {:.2f}\n'.format(
            np.mean(test_accs[-20:]), np.mean(val_accs[-20:])))

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    correct = 0
    total = 0

    labels = torch.zeros(len(test_loader.dataset))
    predicts = torch.zeros(len(test_loader.dataset))

    if not args.no_progress:
        test_loader = tqdm(test_loader)



    TP = torch.zeros(args.num_classes,device=args.device)
    FP = torch.zeros(args.num_classes,device=args.device)
    FN = torch.zeros(args.num_classes,device=args.device)
    TN = torch.zeros(args.num_classes,device=args.device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for b in range(inputs.size(0)):
                labels[indexs[b]]=targets[b]
                predicts[indexs[b]]=predicted[b]

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            tp, fp, fn, tn = f1_score_1(outputs, targets, numclass=args.num_classes)
            TP += tp
            FP += fp
            FN += fn
            TN += tn

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    macroF1 = f1_score(labels, predicts, average="macro")
    acc = 100.*correct/total

    logger.info("top-1 acc: {:.2f}".format(acc))

    return acc, Recall, Precision

def main():
    parser = argparse.ArgumentParser(description='PyTorch T2T Stage2 Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # 数据加载过程中并行处理数据的工作线程数量
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=2400,
                        help='number of labeled data')
    parser.add_argument('--num-unlabeled', type=int, default=16000,
                        help='number of labeled data')
    parser.add_argument('--num-val', type=int, default=3000,
                        help='number of validation data')
    parser.add_argument('--num-ood', type=int, default=4000,
                        help='number of ood data')
    parser.add_argument('--imb-factor', default=50, type=float,
                        help='imbalance ratio of unlabeled data')
    parser.add_argument('--ood-ratio', default=0.6, type=float,
                        help='ood ratio of unlabeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    # warmup epoch:在训练开始时使用较小的学习率，然后逐渐增加学习率，等模型相对稳定后再使用预先的学习率
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    # nesterov momentum:随机梯度下降的变体，通常用于加速训练
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    # mu:未标记数据批量大小相对于标记数据批量大小的比例
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    # 指定模型训练的恢复点
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    # 本地进程的排名
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # 不用进度条来显示训练进度
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--ood-dataset', type=str, default='TIN',
                        choices=['TIN', 'LSUN', 'Gaussian', 'Uniform', '64'],
                        help='choose one dataset as ood data source')
    parser.add_argument('--filter-every-epoch', type=int, default=20,
                        help='every K epoch to filter in distribution unlabeled data')
    # 指定检查点文件的路径
    parser.add_argument('--dg_path', default='cifar10_dg', type=str,
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()  # 解析命令行参数


    if args.seed is not None:
        set_seed(args)

    args.out = "results_fixmatch/cifar_6_4@" + str(args.num_labeled) + "_rou_" + str(int(args.imb_factor)) + \
                "_ood_" + str(args.ood_ratio) + "/seed" + str(args.seed)
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    if args.dataset == 'cifar10':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))

        return model

    args.dg_path = args.dg_path + "/nl_" + str(args.num_labeled) + "_rou_" + \
                    str(int(args.imb_factor)) + "_ood_" + str(args.ood_ratio)
    os.makedirs(args.dg_path, exist_ok=True)
    print("data generate path: {}".format(args.dg_path))
    # the dataset has been dealt with
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')
    print("#ul: {} #val: {} #test: {}".format(len(unlabeled_dataset), len(val_dataset), len(test_dataset)))
    print("#ood: {}".format((unlabeled_dataset.targets > 5).sum()))
    # exit()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size*args.mu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    model = create_model(args)
    model = model.to(args.device)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    train(args, labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, model, ema_model)


if __name__ == '__main__':
    main()
