import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import *
from triplet_loss import *
from data_loader import get_loader
from args import get_parser
from utils.tb_visualizer import Visualizer
from build_vocab import Vocabulary
import pickle
from utils import focal_loss, metrics
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_printoptions(profile="full")

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


G = Generator().to(device)
D = Discriminator().to(device)
cri = nn.BCEWithLogitsLoss().to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)


def main():
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    method = 'SCAN'
    save_folder = method
    os.makedirs(save_folder, exist_ok=True)
    epoch_trace_f_dir = os.path.join(save_folder, "trace_" + method + ".csv")
    with open(epoch_trace_f_dir, "w") as f:
        f.write("epoch,lr,I2R,R@1,R@5,R@10,R2I,R@1,R@5,R@10\n")

    tb_logs = os.path.join('tb_logs', opts.model_name)
    make_dir(tb_logs)
    logger = Visualizer(tb_logs, name='visual_results')

    # models are save only when their loss obtains the best value in the validation
    valtrack = 0
    best_val = float('inf')

    # load models
    image_model = ImageEmbedding().to(device)
    recipe_model = RecipeEmbedding().to(device)

    metric_fc = metrics.ArcMarginProduct(opts.numClasses, opts.numClasses, s=30, m=0.5, easy_margin=False).to(device)

    # load loss functions
    triplet_loss = TripletLoss(device, margin=0.3)
    weight_class = torch.Tensor(opts.numClasses).fill_(1)
    weight_class[0] = 0
    class_criterion = nn.CrossEntropyLoss(weight=weight_class).to(device)
    cross_attn_criterion = ContrastiveLoss()
    mmd = MMDLoss().to(device)
    # focal = focal_loss.FocalLoss().to(device)
    # mkmmd_loss = metrics.MultipleKernelMaximumMeanDiscrepancy(
    #     kernels=[metrics.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
    # ).to(device)
    criterion = [triplet_loss, class_criterion, cross_attn_criterion, mmd]

    # define optimizer
    # 添加L2正则化 , weight_decay=0.001
    optimizer = torch.optim.Adam([
        {'params': image_model.parameters()},
        {'params': recipe_model.parameters()}
    ], lr=opts.lr, betas=(0.5, 0.999))
    steps = 20
    scheduler = lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial image params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial recipe params lr: %f' % optimizer.param_groups[1]['lr'])

    # data preparation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    with open(opts.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # preparing the training loader
    train_loader = get_loader(opts.img_path, train_transform, vocab, opts.data_path, partition='train',
                              batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    val_loader = get_loader(opts.img_path, val_transform, vocab, opts.data_path, partition='test',
                            batch_size=opts.batch_size, shuffle=False, pin_memory=True)
    print('Validation loader prepared.')

    best_val_i2t = {1: 0.0, 5: 0.0, 10: 0.0}
    best_val_t2i = {1: 0.0, 5: 0.0, 10: 0.0}
    best_epoch_i2t = 0
    best_epoch_t2i = 0

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):
        # save current epoch for resuming
        logger.reset()

        # train for one epoch
        train(train_loader, image_model, recipe_model, criterion, optimizer, scheduler, epoch, logger, metric_fc)

        # evaluate on validation set
        recall_i2t, recall_t2i, medR_i2t, medR_t2i = validate(val_loader, image_model, recipe_model, epoch, logger)
        val_loss = (medR_i2t + medR_t2i) / 2

        best_val = min(val_loss, best_val)

        with open(epoch_trace_f_dir, "a") as f:
            lr = optimizer.param_groups[1]['lr']
            f.write("{},{},{},{},{},{},{},{},{},{}\n"
                    .format(epoch, lr, medR_i2t, recall_i2t[1], recall_i2t[5], recall_i2t[10],
                            medR_t2i, recall_t2i[1], recall_t2i[5], recall_t2i[10]))

        for keys in best_val_i2t:
            if recall_i2t[keys] > best_val_i2t[keys]:
                best_val_i2t = recall_i2t
                best_epoch_i2t = epoch + 1
                filename = save_folder + '/recipe_model_best.pkl'
                torch.save(recipe_model.state_dict(), filename)
                break
        for keys in best_val_t2i:
            if recall_t2i[keys] > best_val_t2i[keys]:
                best_val_t2i = recall_t2i
                best_epoch_t2i = epoch + 1
                filename = save_folder + '/image_model_best.pkl'
                torch.save(image_model.state_dict(), filename)
                break

        print("best_i2t: ", best_epoch_i2t, best_val_i2t)
        print("best_t2i: ", best_epoch_t2i, best_val_t2i)
        print('image-lr: %f' % optimizer.param_groups[0]['lr'])
        print('recipe-lr: %f' % optimizer.param_groups[1]['lr'])
        print(' ')

    logger.close()


def train(train_loader, image_model, recipe_model, criterion, optimizer, scheduler, epoch, logger, metric_fc):
    batch_time = AverageMeter()
    tri_losses = AverageMeter()
    img_losses = AverageMeter()
    rec_losses = AverageMeter()
    sca_losses = AverageMeter()
    mmd_losses = AverageMeter()
    gan_losses = AverageMeter()

    total_loss_dict = {
        'tri_loss': [],
        'img_loss': [],
        'rec_loss': [],
        'sca_loss': [],
        'mmd_loss': [],
        'gan_loss': [],
    }

    label = list(range(0, opts.batch_size))
    label.extend(label)
    label = np.array(label)
    label = torch.tensor(label).to(device).long()

    # switch to train mode
    image_model.train()
    recipe_model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # data: [images, instrs, instr_len, ingrs, ingr_len, list(food_id)], \
        #            [targets, lengths, class_label, ret, one_hot_vec]
        inputs_var = list()
        for j in range(len(inputs)-1):
            inputs_var.append(inputs[j].to(device))
        targets_var = list()
        for j in range(len(targets)):
            targets_var.append(targets[j].to(device))

        image_embedding = image_model(inputs_var[0])
        recipe_embedding = recipe_model(inputs_var[1], inputs_var[2], inputs_var[3], inputs_var[4])

        # cross-modal retrieval
        # L_Ret
        tri_loss = global_loss(criterion[0], torch.cat([image_embedding[0], recipe_embedding[0]]), label)[0]

        # translation consisteny
        # L_Trans
        img_class_loss = criterion[1](image_embedding[1], targets_var[2])
        rec_class_loss = criterion[1](recipe_embedding[1], targets_var[2])

        # # ArcFace loss
        # img_class = metric_fc(image_embedding[1], targets_var[2])
        # rec_class = metric_fc(recipe_embedding[1], targets_var[2])
        # img_arc_loss = criterion[4](img_class, targets_var[2])
        # rec_arc_loss = criterion[4](rec_class, targets_var[2])

        # KL_loss
        img_logits = image_embedding[1]
        rec_logits = recipe_embedding[1]
        img_probs = F.softmax(img_logits, dim=1)
        rec_probs = F.softmax(rec_logits, dim=1)
        img_log_probs = F.log_softmax(img_logits, dim=1)
        rec_log_probs = F.log_softmax(rec_logits, dim=1)
        KL_loss_img = F.kl_div(img_log_probs, rec_probs.detach(), reduction='sum') / img_log_probs.shape[0]
        KL_loss_rec = F.kl_div(rec_log_probs, img_probs.detach(), reduction='sum') / rec_log_probs.shape[0]

        img_loss = img_class_loss + KL_loss_rec
        rec_loss = rec_class_loss + KL_loss_img
        # img_loss = img_class_loss
        # rec_loss = rec_class_loss

        # Cross Attention Loss
        sca_loss_i2t_instr = criterion[2](image_embedding[2], recipe_embedding[2], inputs_var[2])
        sca_loss_t2i_instr = criterion[2](image_embedding[2], recipe_embedding[2], inputs_var[2])
        sca_loss_i2t_ingr = criterion[2](image_embedding[2], recipe_embedding[3], inputs_var[4])
        sca_loss_t2i_ingr = criterion[2](image_embedding[2], recipe_embedding[3], inputs_var[4])
        sca_loss = (sca_loss_i2t_instr + sca_loss_t2i_instr) / 2 + (sca_loss_i2t_ingr + sca_loss_t2i_ingr) / 2

        # mmd_loss = criterion[3](image_embedding[0], recipe_embedding[0])
        # mmd_loss_1 = criterion[3](image_embedding[0], recipe_embedding[0])
        # mmd_loss_2 = criterion[3](image_embedding[1], recipe_embedding[1])
        mmd_loss_3 = criterion[3](recipe_embedding[0], image_embedding[0])
        mmd_loss_4 = criterion[3](recipe_embedding[1], image_embedding[1])
        mmd_loss = mmd_loss_3 + mmd_loss_4

        # CGAN loss
        realscore = torch.ones(opts.batch_size, 1).to(device)
        fakescore = torch.zeros(opts.batch_size, 1).to(device)

        # 训练鉴别器————总的损失为两者相加
        # image_embedding[1] = image_embedding[1].long()
        fakeimage = G(image_embedding[0], image_embedding[1])
        d_realimage_loss = cri(D(targets_var[3], image_embedding[1]), realscore)
        d_fakeimage_loss = cri(D(fakeimage, image_embedding[1]), fakescore)
        D_loss = d_realimage_loss + d_fakeimage_loss
        D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        # 训练生成器————损失只有一个
        fakeimage = G(image_embedding[0], image_embedding[1])
        G_loss = cri(D(fakeimage, image_embedding[1]), realscore)
        G_optimizer.zero_grad()
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        gan_loss = -torch.mean(D_loss)

        # combined loss
        # loss = tri_loss + 0.02 * mmd_loss + 0.02 * sca_loss + 0.02 * ((img_loss + rec_loss) / 2)
        # loss = tri_loss + 0.05 * mmd_loss + 0.02 * ((img_loss + rec_loss) / 2)
        # loss = tri_loss + 0.02 * mmd_loss + 0.02 * ((img_loss + rec_loss) / 2) + 0.02 * gan_loss
        loss = tri_loss + 0.01 * mmd_loss + 0.005 * gan_loss

        batch_size = inputs_var[0].data.cpu().size(0)
        tri_losses.update(tri_loss.item(), batch_size)
        img_losses.update(img_loss.item(), batch_size)
        rec_losses.update(rec_loss.item(), batch_size)
        sca_losses.update(sca_loss.item(), batch_size)
        mmd_losses.update(mmd_loss.item(), batch_size)
        gan_losses.update(gan_loss.item(), batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    scheduler.step()

    print('Epoch: {0}  '
          'tri loss {tri_loss.val:.4f} ({tri_loss.avg:.4f}),  '
          'img loss {img_loss.val:.4f} ({img_loss.avg:.4f}),  '
          'rec loss {rec_loss.val:.4f} ({rec_loss.avg:.4f}),  '
          'sca loss {sca_loss.val:.4f} ({sca_loss.avg:.4f}),  '
          'mmd loss {mmd_loss.val:.4f} ({mmd_loss.avg:.4f}),  '
          'gan loss {gan_loss.val:.4f} ({gan_loss.avg:.4f})'
          .format(epoch+1, tri_loss=tri_losses, img_loss=img_losses, rec_loss=rec_losses,
                  sca_loss=sca_losses, mmd_loss=mmd_losses, gan_loss=gan_losses))

    total_loss_dict['tri_loss'] = tri_losses.val
    total_loss_dict['img_loss'] = img_losses.val
    total_loss_dict['rec_loss'] = rec_losses.val
    total_loss_dict['sca_loss'] = sca_losses.val
    total_loss_dict['mmd_loss'] = mmd_losses.val
    total_loss_dict['gan_loss'] = gan_losses.val
    logger.scalar_summary(
        mode='train',
        epoch=epoch+1,
        **{k: v for k, v in total_loss_dict.items() if v}
    )


def validate(val_loader, image_model, recipe_model, epoch, logger):
    total_loss_dict = {
        'medR_i2t': [],
        'medR_t2i': [],
        'top1_i2t': [],
        'top5_i2t': [],
        'top10_i2t': [],
        'top1_t2i': [],
        'top5_t2i': [],
        'top10_t2i': []
    }

    # switch to evaluate mode
    image_model.eval()
    recipe_model.eval()

    for i, (inputs, targets) in enumerate(val_loader):
        # data: [images, instrs, instr_len, ingrs, ingr_len, list(food_id)], \
        #            [targets, lengths, class_label, ret, one_hot_vec]
        inputs_var = list()
        for j in range(len(inputs) - 1):
            inputs_var.append(inputs[j].to(device))
        targets_var = list()
        for j in range(len(targets)):
            targets_var.append(targets[j].to(device))

        with torch.no_grad():
            image_embedding = image_model(inputs_var[0])
            recipe_embedding = recipe_model(inputs_var[1], inputs_var[2], inputs_var[3], inputs_var[4])

            if i == 0:
                data0 = image_embedding[0].data.cpu().numpy()
                data1 = recipe_embedding[0].data.cpu().numpy()
            else:
                data0 = np.concatenate((data0, image_embedding[0].data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, recipe_embedding[0].data.cpu().numpy()), axis=0)

    medR_i2t, recall_i2t = rank(data0, data1, 'image')
    print('I2T Val medR {medR:.4f}\t Recall {recall}'.format(medR=medR_i2t, recall=recall_i2t))

    medR_t2i, recall_t2i = rank(data0, data1, 'recipe')
    print('T2I Val medR {medR:.4f}\t Recall {recall}'.format(medR=medR_t2i, recall=recall_t2i))

    total_loss_dict['medR_i2t'] = medR_i2t
    total_loss_dict['medR_t2i'] = medR_t2i
    total_loss_dict['top1_i2t'] = recall_i2t[1]
    total_loss_dict['top5_i2t'] = recall_i2t[5]
    total_loss_dict['top10_i2t'] = recall_i2t[10]
    total_loss_dict['top1_t2i'] = recall_t2i[1]
    total_loss_dict['top5_t2i'] = recall_t2i[5]
    total_loss_dict['top10_t2i'] = recall_t2i[10]
    logger.scalar_summary(
        mode='val',
        epoch=epoch+1,
        **{k: v for k, v in total_loss_dict.items() if v}
    )

    return recall_i2t, recall_t2i, medR_i2t, medR_t2i


def rank(img_embeds, rec_embeds, type_embedding='image'):
    random.seed(opts.seed)
    img_vec = img_embeds
    instr_vec = rec_embeds

    # Ranker
    N = opts.medR
    idxs = range(N)

    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(10):
        # 找到N个样本
        ids = random.sample(range(0, len(img_embeds)), N)  # 多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
        img_sub = img_vec[ids, :]
        instr_sub = instr_vec[ids, :]

        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        if type_embedding == 'image':
            for ii in idxs:
                distance = {}
                for j in range(N):
                    distance[j] = np.linalg.norm(img_sub[ii] - instr_sub[j])    # 求第二范数
                distance_sorted = sorted(distance.items(), key=lambda x: x[1])
                pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

                if (pos + 1) == 1:
                    recall[1] += 1
                if (pos + 1) <= 5:
                    recall[5] += 1
                if (pos + 1) <= 10:
                    recall[10] += 1

                # store the position
                med_rank.append(pos + 1)
        else:
            for ii in idxs:
                distance = {}
                for j in range(N):
                    distance[j] = np.linalg.norm(instr_sub[ii] - img_sub[j])  # 求第二范数
                distance_sorted = sorted(distance.items(), key=lambda x: x[1])
                pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

                if (pos + 1) == 1:
                    recall[1] += 1
                if (pos + 1) <= 5:
                    recall[5] += 1
                if (pos + 1) <= 10:
                    recall[10] += 1

                # store the position
                med_rank.append(pos + 1)

        med = np.median(med_rank)
        glob_rank.append(med)

        for j in recall.keys():
            recall[j] = recall[j] / N
            glob_recall[j] += recall[j]

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10

    return np.average(glob_rank), glob_recall


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0).to(device),
                                   requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,  # fake samples
        inputs=interpolates,  # real samples
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


if __name__ == '__main__':
    main()
