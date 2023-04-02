from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from args import get_parser
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


def norm(x, p=2, dim=1, eps=1e-12):
    return x / x.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(x)


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'.
    """
    def __init__(self, device, margin=None):
        self.margin = margin
        self.device = device
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):

        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1)).to(self.device)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # import pdb; pdb.set_trace()
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze( 0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels, normalize_feature=False):
    # tri_loss: nn.MarginRankingLoss(margin=margin)
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)   # torch.Size([8, 1024])
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)     # torch.Size([8, 8])
    # torch.Size([8]), torch.Size([8])
    dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an, dist_mat


def func_attention(query, context, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    # print(context.shape)
    # print(queryT.shape)
    attn = torch.bmm(context, queryT)
    if opts.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opts.raw_feature_norm == "l2norm":
        attn = norm(attn, 2)
    elif opts.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = norm(attn, 2)
    elif opts.raw_feature_norm == "l1norm":
        attn = norm(attn, 2)
    elif opts.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = norm(attn, 2)
    elif opts.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opts.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opts.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)

    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    return w12 / (w1 * w2).clamp(min=eps)


def xattn_score_t2i(images, recipes, cap_lens):
    """
    Images: (n_image, n_regions, d) matrix of images
    Recipes: (n_caption, max_n_word, d) matrix of recipes
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_recipe = recipes.size(0)
    for i in range(n_recipe):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = recipes[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, smooth=opts.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opts.agg_func == 'LogSumExp':
            row_sim.mul_(opts.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opts.lambda_lse
        elif opts.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opts.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opts.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opts.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, recipes, cap_lens):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Recipes: (batch_size, max_n_words, d) matrix of recipes
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_recipe = recipes.size(0)
    n_region = images.size(1)
    for i in range(n_recipe):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = recipes[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        # 在第一个维度上, 扩充n_image倍
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, smooth=opts.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opts.agg_func == 'LogSumExp':
            row_sim.mul_(opts.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opts.lambda_lse
        elif opts.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opts.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opts.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opts.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    # attn --> (batch, sourceL, queryL)
    # positive attention
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)
    attn_pos = l1norm(attn_pos, 1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)

    return  attn_pos


def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn


def xattn_score(images, captions, cap_lens, opt):
    """
    Note that this function is used to train the model with Discriminative Mismatch Mining.
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    max_pos = []
    max_neg = []
    max_pos_aggre = []
    max_neg_aggre = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    cap_len_i = torch.zeros(1, n_caption)
    n_region = images.size(1)
    batch_size = n_image
    N_POS_WORD = 0
    A = 0
    B = 0
    mean_pos = 0
    mean_neg = 0

    for i in range(n_caption):

        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_len_i[0, i] = n_word
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # text-to-image direction
        t2i_sim = torch.zeros(batch_size * n_word).double().to(device)
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)

        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * opt.thres

        # --------------------------------------------------------------------------------------------------------------------------
        # Neg-Pos Branch Matching
        # negative attention
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
        attn_neg = Row_max.lt(0).float()
        t2i_sim_neg = Row_max * attn_neg
        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
        weiContext_pos = torch.bmm(attn_pos, images)
        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r

        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)
        # --------------------------------------------------------------------------------------------------------------------------

        # Discriminative Mismatch Mining
        # --------------------------------------------------------------------------------------------------------------------------
        wrong_index = sim.sort(0, descending=True)[1][0].item()
        # Based on the correctness of the calculated similarity ranking,
        # we devise to decide whether to update at each sampling time.
        if wrong_index == i:
            # positive samples
            attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), 1)[0].to(device)
            attn_max_row_pos = attn_max_row[(i * n_word): (i * n_word + n_word)].to(device)

            # negative samples
            neg_index = sim.sort(0)[1][0].item()
            attn_max_row_neg = attn_max_row[(neg_index * n_word): (neg_index * n_word + n_word)].to(device)

            max_pos.append(attn_max_row_pos)
            max_neg.append(attn_max_row_neg)
            N_POS_WORD = N_POS_WORD + n_word
            if N_POS_WORD > 200:  # 200 is the empirical value to make adequate samplings
                max_pos_aggre = torch.cat(max_pos, 0)
                max_neg_aggre = torch.cat(max_neg, 0)
                mean_pos = max_pos_aggre.mean().to(device)
                mean_neg = max_neg_aggre.mean().to(device)
                stnd_pos = max_pos_aggre.std()
                stnd_neg = max_neg_aggre.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)
                B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(
                    2) * torch.log(stnd_neg / (opt.alpha * stnd_pos) + 1e-8)

                thres = opt.thres
                thres_safe = opt.thres_safe
                opt.stnd_pos = stnd_pos.item()
                opt.stnd_neg = stnd_neg.item()
                opt.mean_pos = mean_pos.item()
                opt.mean_neg = mean_neg.item()

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    # A more simple way to calculate the learning boundary after alpha* adjustement
                    # In implementation, we can use a more feasible opt.thres_safe,
                    # i.e. directly calculate the empirical lower bound, as in the Supplementary Material.
                    # (note that alpha* theoretically unifies the opt.thres at training
                    # and opt.thres_safe at testing into the same concept)
                    opt.thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
                    opt.thres_safe = (mean_pos - 3 * opt.stnd_pos).item()

                if opt.thres < 0:
                    opt.thres = 0
                if opt.thres > 1:
                    opt.thres = 0

                if opt.thres_safe < 0:
                    opt.thres_safe = 0
                if opt.thres_safe > 1:
                    opt.thres_safe = 0

                opt.thres = 0.7 * opt.thres + 0.3 * thres
                opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe

        if N_POS_WORD < 200:
            opt.thres = 0
            opt.thres_safe = 0

        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images, recipes, cap_lens, op='image', ty='region'):
        # scores = xattn_score(images, recipes, cap_lens, opts)
        if ty == 'region':
            if op == 'image':
                scores = xattn_score_i2t(images, recipes, cap_lens)
            else:
                scores = xattn_score_t2i(recipes, images, cap_lens)
        else:
            scores = order_sim(images, recipes)

        diagonal = scores.diag().view(images.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_image = (self.margin + scores - d2).clamp(min=0)
        # compare every diagonal score to scores in its column
        # recipe retrieval
        cost_recipe = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(device)
        cost_image = cost_image.masked_fill_(I, 0)
        cost_recipe = cost_recipe.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_image = cost_image.max(0)[0]       # torch.Size([4, 4])
            cost_recipe = cost_recipe.max(1)[0]     # torch.Size([4, 4])

        return cost_image.sum() + cost_recipe.sum()


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
