import os
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
# cifar10_mean = (0.5, 0.5, 0.5)
# cifar10_std = (0.5, 0.5, 0.5)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    # for train
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    # for test
    test_dataset = datasets.CIFAR10(root, train=False, download=True)

    # move class "plane" and "car" to label 8 and 9
    base_dataset.targets = (np.array(base_dataset.targets) - 2)
    test_dataset.targets = (np.array(test_dataset.targets) - 2)
    (base_dataset.targets)[np.where((base_dataset.targets) == -2)] = 8
    (base_dataset.targets)[np.where((base_dataset.targets) == -1)] = 9
    (test_dataset.targets)[np.where((test_dataset.targets) == -2)] = 8
    (test_dataset.targets)[np.where((test_dataset.targets) == -1)] = 9

    # index
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, ori_labeled_idx = x_u_v_split(args, base_dataset.targets)
    test_idxs = test_split(args, test_dataset.targets)

    print('label_len: {} unlabel_len: {} val_len: {} test_len: {}'.format(len(ori_labeled_idx), len(train_unlabeled_idxs), len(val_idxs),len(test_idxs)))
    labeled_dataset = {}
    labeled_data = (base_dataset.data)[train_labeled_idxs]
    labeled_target = (base_dataset.targets)[train_labeled_idxs]
    labeled_dataset['images'] = labeled_data
    labeled_dataset['labels'] = labeled_target

    unlabeled_dataset = {}
    unlabeled_data = (base_dataset.data)[train_unlabeled_idxs]
    unlabeled_target = (base_dataset.targets)[train_unlabeled_idxs]
    unlabeled_dataset['images'] = unlabeled_data
    unlabeled_dataset['labels'] = unlabeled_target

    labeled_ori_dataset = {}
    labeled_ori_data = (base_dataset.data)[ori_labeled_idx]
    labeled_ori_target = (base_dataset.targets)[ori_labeled_idx]
    labeled_ori_dataset['images'] = labeled_ori_data
    labeled_ori_dataset['labels'] = labeled_ori_target


    val_dataset = {}
    val_data = (base_dataset.data)[val_idxs]
    val_target = (base_dataset.targets)[val_idxs]
    val_dataset['images'] = val_data
    val_dataset['labels'] = val_target

    test_da = {}
    test_data = (test_dataset.data)[test_idxs]
    test_target = (test_dataset.targets)[test_idxs]
    test_da['images'] = test_data
    test_da['labels'] = test_target


 
    train_labeled_dataset = CIFAR10_SSL(labeled_dataset, transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10_SSL(unlabeled_dataset, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    val_dataset = CIFAR10_SSL(val_dataset, transform=transform_val)

    test_da = CIFAR10_SSL(test_da, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_da


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class CIFAR10_SSL(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.images = self.dataset['images']
        self.targets = self.dataset['labels']
    def __len__(self):
        return len((self.dataset)['images'])

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)       # TransformFixMatch.__call__()

        return img, target, index

def x_u_v_split(args, labels):
    if os.path.exists(args.dg_path + "/label_idx.npy"):
        labeled_idx = np.load(args.dg_path + "/label_idx.npy")
        val_idx = np.load(args.dg_path + "/val_idx.npy")
        unlabeled_id_idx = np.load(args.dg_path + "/unlabeled_id_idx.npy")
        unlabeled_ood_idx = np.load(args.dg_path + "/ood_idx.npy")
        unlabeled_idx = unlabeled_id_idx.tolist()
        unlabeled_idx.extend(unlabeled_ood_idx.tolist())
        unlabeled_idx = np.array(unlabeled_idx)
        print("================load dataset successfully!================")
    else:
        val_per_class = args.num_val // args.num_classes
        label_per_class = args.num_labeled // args.num_classes

        img_unlabel_list = get_img_num_per_cls(args, args.dataset, args.imb_factor, label_per_class)
        unlabel_id_np = np.array(img_unlabel_list)
        np.save(args.dg_path + "/ul_id_dt_np.npy", unlabel_id_np)
        print("unlabel id distribution: {}".format(img_unlabel_list))

        img_ul_total = (np.array(img_unlabel_list)).sum()

        num_ood = int((img_ul_total * args.ood_ratio) / (1 - args.ood_ratio))

        labels = np.array(labels)
        labeled_idx = []
        val_idx = []
        unlabeled_idx = []

        unlabeled_ood_per_class = num_ood // (10 - args.num_classes)

        for i in range(args.num_classes):
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            unlabeled_per_class = img_unlabel_list[i]
            labeled_idx.extend(idx[:label_per_class])
            unlabeled_idx.extend(idx[label_per_class:label_per_class+unlabeled_per_class])
            val_idx.extend(idx[-val_per_class:])
        unlabel_id_idx = np.array(unlabeled_idx)
        np.save(args.dg_path + "/unlabeled_id_idx.npy", unlabel_id_idx)
        # print(len(unlabeled_idx))
        ood_idx = []
        for i in range(args.num_classes, 10):
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            # unlabeled_idx.extend(idx[:unlabeled_ood_per_class])
            ood_idx.extend(idx[:unlabeled_ood_per_class])
        unlabel_ood_idx = np.array(ood_idx)
        np.save(args.dg_path + "/ood_idx.npy", unlabel_ood_idx)
        unlabeled_idx.extend(ood_idx)
        labeled_idx = np.array(labeled_idx)
        np.save(args.dg_path + "/label_idx.npy",labeled_idx)
        val_idx = np.array(val_idx)
        np.save(args.dg_path + "/val_idx.npy", val_idx)
        unlabeled_idx = np.array(unlabeled_idx)
        print("=======labled========")
        print(labeled_idx)
        print("labeled length: {}".format(len(labeled_idx)))
        print("========val========")
        print(val_idx)
        print("validation length: {}".format(len(val_idx)))
        print("========unlabled========")
        print(unlabeled_idx)
        print("unlabeled id length: {}".format(len(unlabeled_idx)))

    assert len(labeled_idx) == args.num_labeled
    assert len(val_idx) == args.num_val
    ori_labeled_idx = labeled_idx
    unlabeled_ori_idx = unlabeled_idx

    # confirm the selected data is fixed(run twice)
    print('======================1======================')
    print(ori_labeled_idx)
    print('======================2======================')
    print(unlabeled_ori_idx)
    print('======================3======================')
    print(len(unlabeled_idx))
    # exit()
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_ori_idx, val_idx, ori_labeled_idx

def test_split(args, labels):
    test_idxs = []

    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:])

    np.random.shuffle(test_idxs)

    return test_idxs

def get_img_num_per_cls(args, dataset, imb_factor, label_per_class):
    if dataset == 'cifar10':
        img_max = 45000/10
        img_max = img_max - label_per_class
        cls_num = 6


    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * ((1. / imb_factor)**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


DATASET_GETTERS = {'cifar10': get_cifar10}
