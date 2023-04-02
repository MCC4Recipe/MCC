from __future__ import print_function

import sys

import torch.utils.data as data
from PIL import Image
import os
import pickle
import numpy as np
import lmdb
import torch
import torchvision.transforms as transforms
from args import get_parser

parser = get_parser()
opts = parser.parse_args()


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, data_path, partition, transform=None, vocab=None):
        with open(os.path.join(data_path + partition + '_ids.pkl'), 'rb') as f:
            self.ids = pickle.load(f, encoding='latin1')
        with open(os.path.join(data_path + partition + '_split.pkl'), 'rb') as f:
            self.split = pickle.load(f, encoding='latin1')

        self.env = lmdb.open(os.path.join(img_path, partition + '_lmdb'),
                             max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.image_file = lmdb.open(os.path.join("../im2recipe/imgdata", 'lmdb_' + partition),
                                    max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        self.partition = partition
        self.transform = transform
        self.vocab = vocab

        self.img_path = img_path
        self.maxInst = 20

    def __getitem__(self, index):
        # for background
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
        sample = pickle.loads(serialized_sample, encoding='latin1')
        # print(sample): ingrs; imgs: url, id; instrs
        imgs = sample['imgs']
        food_id = self.ids[index]

        if self.partition != 'train':
            # 对于训练集, 每个recipe只取前五张图片
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            # 对于测试集和验证集, 只取第一张图片
            imgIdx = 0
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)  # *parameters是用来接受任意多个参数并将其放在一个元组中
        # path = os.path.join(self.img_path, self.partition, loader_path, imgs[imgIdx]['id'])

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # images
        try:
            # img = Image.open(path).convert('RGB')
            with self.image_file.begin(write=False) as txn:
                img = txn.get(imgs[imgIdx]['id'].encode())
                img = np.fromstring(img, dtype=np.uint8)
                img = np.reshape(img, (256, 256, 3))
            img = Image.fromarray(img.astype('uint8'), 'RGB')
        except:
            # print(..., file=sys.stderr)
            img = Image.new('RGB', (256, 256), 'white')

        img = self.transform(img)
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        re_img = transforms.Resize(128)(img)
        img = normalize(img)
        ret = normalize(re_img)

        class_label = sample['classes'] - 1

        vocab = self.vocab
        selec_ingrs = set(self.split[food_id]['ingredients'])
        ingr_cap = []
        ingr_cap.append(vocab('<start>'))

        one_hot_vec = torch.zeros(4102)
        for i in list(selec_ingrs):
            one_hot_vec[vocab(str(i).lower())] = 1

        ingr_cap = torch.Tensor(ingr_cap)

        # output
        # also output the length of captions, which could be used in LSTM prediction
        return img, instrs, itr_ln, ingrs, igr_ln, \
               ingr_cap, class_label, ret, one_hot_vec, food_id

    def __len__(self):
        # return len(self.ids)
        return int(len(self.ids) / 10)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).

    data.sort(key=lambda x: len(x[5]), reverse=True)
    img, instrs, itr_ln, ingrs, igr_ln, \
    ingr_cap, class_label, ret, one_hot_vec, food_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(img, 0)
    instrs = torch.stack(instrs, 0)
    itr_ln = torch.LongTensor(list(itr_ln))
    ingrs = torch.stack(ingrs, 0)
    igr_ln = torch.LongTensor(list(igr_ln))
    class_label = torch.LongTensor(list(class_label))
    ret = torch.stack(ret, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in ingr_cap]
    targets = torch.zeros(len(ingr_cap), max(lengths)).long()
    for i, cap in enumerate(ingr_cap):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    one_hot_vec = torch.stack(one_hot_vec, 0)

    # list不能to(device), 需要先用torch.tensor转换
    return [images, instrs, itr_ln, ingrs, igr_ln, list(food_id)], \
           [targets, torch.tensor(lengths), class_label, ret, one_hot_vec]


def get_loader(img_path, transform, vocab, data_path, partition, batch_size, shuffle, num_workers=0, pin_memory=False):
    data_loader = torch.utils.data.DataLoader(ImagerLoader(img_path, data_path, partition, transform, vocab),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              drop_last=True,
                                              collate_fn=collate_fn)
    return data_loader
