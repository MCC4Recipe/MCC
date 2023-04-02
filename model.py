import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torchvision.models as models
# import torchwordemb
from utils import wordemb
from torch.autograd import Variable
import torch.nn.functional as F
from args import get_parser
import numpy as np
from math import sqrt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


fc_sia = nn.Sequential(
    nn.Linear(opts.embDim, opts.embDim),
    nn.BatchNorm1d(opts.embDim),
    nn.Tanh(),
).to(device)


def norm(x, p=2, dim=1, eps=1e-12):
    return x / x.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(x)


class AttentionImage(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        # return norm(self.gamma * out + input)
        return self.fc(self.gamma * out + input)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttn(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SelfAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(SelfAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.soft_max = nn.Softmax(dim=2)

    def forward(self, q):
        attn = torch.bmm(q, q.transpose(1, 2))
        mask = attn.eq(0.0)     # 相等则为1
        attn = attn / self.temperature
        # 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。
        attn = attn.masked_fill(mask, -np.inf)
        # 用softmax函数将N维输入进行归一化，归一化之后每个输出的Tensor范围在[0, 1]，并且归一化的那一维和为1
        attn = self.soft_max(attn)
        attn = attn.masked_fill(attn != attn, 0.0)
        attn = self.dropout(attn)
        output = torch.bmm(attn, q)

        return output, attn


# Skip-thoughts LSTM
class InstrRNN(nn.Module):
    def __init__(self):
        super(InstrRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)

        self.attention = SelfAttention(temperature=np.power(opts.srnnDim, 0.5))

    def forward(self, x, sq_lengths):
        # print(x.shape)  # torch.Size([64, 20, 1024])
        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)
        # print(out[0].shape)     # torch.Size([252, 1024])
        # print(hidden[0].shape)  # torch.Size([1, 64, 1024])

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
        # print(unpacked.size())  # torch.Size([*, 32, 1024])
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(unpacked)

        each_hidden_out = unpacked.gather(1, unsorted_idx).transpose(0, 1).contiguous()
        # print(each_hidden_out.size())   # torch.Size([32, *, 1024])

        output, attn = self.attention(each_hidden_out)
        layer_norm = nn.LayerNorm(each_hidden_out.size()[1:]).to(device)
        output = layer_norm(output + each_hidden_out)

        return output


class IngrRNN(nn.Module):
    def __init__(self):
        super(IngrRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.embDim, hidden_size=opts.embDim, bidirectional=True, batch_first=True)
        _, vec = wordemb.load_embedding("./utils/instr.vec")
        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)  # not sure about the padding idx
        self.embs.weight.data.copy_(vec)

        self.ingr_embedding = nn.Sequential(
            nn.Linear(opts.ingrW2VDim, opts.embDim),
            nn.Tanh()
        )
        self.self_attention = SelfAttention(temperature=np.power(opts.embDim, 0.5))

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)  # torch.Size([64, 20, 300])
        x = self.ingr_embedding(x)

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)
        # print(out[0].shape)     # torch.Size([228, 600])
        # print(hidden[0].shape)  # torch.Size([2, 64, 300])
        each_hidden_out = torch.nn.utils.rnn.pad_packed_sequence(out)[0]

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(each_hidden_out)
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        each_hidden_out = each_hidden_out.gather(1, unsorted_idx).transpose(0, 1).contiguous()

        each_hidden_out = (each_hidden_out[:, :, :each_hidden_out.size(2) // 2] +
                           each_hidden_out[:, :, each_hidden_out.size(2) // 2:]) / 2

        # SelfAttention + LayerNorm
        output, attn = self.self_attention(each_hidden_out)
        layer_norm = nn.LayerNorm(each_hidden_out.size()[1:]).to(device)
        output = layer_norm(output + each_hidden_out)

        return output


class RecipeEmbedding(nn.Module):
    def __init__(self):
        super(RecipeEmbedding, self).__init__()
        self.instrRNN = InstrRNN()
        self.ingrRNN = IngrRNN()

        self.recipe_embedding = nn.Sequential(
            nn.Linear(opts.embDim + opts.srnnDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh()
        )
        self.fc_recipe = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh()
        )

        self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, y1, y2, z1, z2):
        # instrs, itr_ln, ingrs, igr_ln
        instr_cap = self.instrRNN(y1, y2)   # torch.Size([32, 13, 1024])
        ingr_cap = self.ingrRNN(z1, z2)     # torch.Size([32, 9, 600])

        instr_emb = torch.mean(instr_cap.transpose(1, 2), 2)    # torch.Size([32, 1024])
        ingr_emb = torch.mean(ingr_cap.transpose(1, 2), 2)      # torch.Size([32, 600])
        recipe_emb = torch.cat([instr_emb, ingr_emb], dim=1)    # joining on the last dim
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)
        recipe_emb = fc_sia(recipe_emb)
        # recipe_emb = self.fc_recipe(recipe_emb)
        recipe_emb_norm = norm(recipe_emb)

        recipe_sem = self.semantic_branch(recipe_emb)

        output = [recipe_emb_norm, recipe_sem, instr_cap, ingr_cap]

        return output


class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        resnet = models.resnet50(pretrained=True)
        print(resnet)
        modules = list(resnet.children())[:-2]  # we do not use the last fc layer.
        self.visionMLP = nn.Sequential(*modules)
        self.conv = nn.Sequential(
            nn.Conv2d(opts.imfeatDim, opts.imfeatDim, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.inplanes = opts.imfeatDim
        self.attn_image = AttentionImage(self.inplanes)
        self.coord_attn = CoordAttn(self.inplanes, self.inplanes, 16)
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.gem_pool = GeneralizedMeanPooling(output_size=(1, 1))
        self.self_attn = SelfAttention(temperature=np.power(opts.embDim, 0.5))

        self.region_embedding = nn.Sequential(
            nn.Linear(self.inplanes, opts.embDim),
            nn.BatchNorm1d(7*7),
            nn.Tanh()
        )
        self.visual_embedding = nn.Sequential(
            nn.Linear(self.inplanes, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh()
        )
        self.fc_visual = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh()
        )

        self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x):
        # print(x.shape)        # torch.Size([32, 3, 224, 224])
        x = self.visionMLP(x)   # torch.Size([32, 2048, 7, 7])
        # x = self.conv(x)        # torch.Size([32, 2048, 7, 7])
        # x = self.attn_image(x)
        # x = self.coord_attn(x)
        x = self.ca(x) * x
        x = self.sa(x) * x

        region_emb = x.view(x.size(0), opts.imfeatDim, -1)
        region_emb = region_emb.transpose(1, 2)  # torch.Size([32, 49, 1024])
        region_emb = self.region_embedding(region_emb)
        region_emb, _ = self.self_attn(region_emb)

        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        visual_emb = torch.cat([avg_pool + max_pool], dim=1)
        # visual_emb = self.gem_pool(x)
        # visual_emb = self.avg_pool(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)  # batch_size * 1024
        visual_emb = self.visual_embedding(visual_emb)
        # visual_emb = visual_emb + torch.mean(region_emb.transpose(1, 2), 2).squeeze()
        visual_emb = torch.cat([visual_emb + torch.mean(region_emb.transpose(1, 2), 2).squeeze()], dim=-1)
        visual_emb = norm(visual_emb)
        visual_emb = fc_sia(visual_emb)
        # visual_emb = self.fc_visual(visual_emb)
        visual_emb_norm = norm(visual_emb)

        visual_sem = self.semantic_branch(visual_emb_norm)

        output = [visual_emb_norm, visual_sem, region_emb]

        return output


class MMDLoss(nn.Module):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


img_shape = (3, 128, 128)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opts.numClasses, opts.numClasses)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opts.embDim + opts.numClasses, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        # label = self.label_emb(labels)
        gen_input = torch.cat((labels, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opts.numClasses, opts.numClasses)

        self.model = nn.Sequential(
            nn.Linear(opts.numClasses + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # self.label_embedding(labels)
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

