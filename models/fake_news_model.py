import torch
import torch.nn.functional as F
from torch import nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
import numpy as np
from .attention import Dot_Attention

class ProjectionHead(nn.Module):
    def __init__(
            self,
            config,
            embedding_dim,
    ):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(embedding_dim, config.projection_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config.projection_size, config.projection_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.projection_size)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(4 * config.projection_size)
        self.linear_layer = nn.Linear(4 * config.projection_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(config.dropout)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.classifier_layer = nn.Linear(config.hidden_size, config.class_num)
        self.softmax = nn.Softmax(dim=1)
        self.Relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.linear_layer(x)
        x = self.gelu(x)
        x = self.drop_out(x)
        self.embeddings = x = self.layer_norm_2(x)
        x = self.classifier_layer(x)
        x = self.softmax(x) 
        return x


class text_classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_norm = nn.LayerNorm(input_dim * 2)
        self.linear_layer = nn.Linear(input_dim, input_dim * 2)
        self.fc = nn.Linear(input_dim * 2, input_dim * 2)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(0.5)
        self.classifier_layer = nn.Linear(input_dim * 2, output_dim)
        self.soft_plus = nn.Softplus()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.drop_out(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.classifier_layer(x)
        x = self.soft_plus(x)
        return x


class image_classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_norm = nn.LayerNorm(input_dim * 2)
        self.linear_layer = nn.Linear(input_dim, input_dim * 2)
        self.fc = nn.Linear(input_dim * 2, input_dim * 2)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(0.5)
        self.classifier_layer = nn.Linear(input_dim * 2, output_dim)
        self.soft_plus = nn.Softplus()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.drop_out(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.classifier_layer(x)
        x = self.soft_plus(x)
        return x


class FakeNewsModel(nn.Module):
    def __init__(
            self, config
    ):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        class_weights = torch.FloatTensor(config.class_weights)
        self.classifier_loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

        self.text_embeddings = None
        self.image_embeddings = None
        self.multimodal_embeddings = None

        self.scaler = 1.0

        self.text_classifier = text_classifier(768, 2)
        self.image_classifier = image_classifier(768, 2)

        self.image_encoder_teacher = ImageEncoder(config)
        self.text_encoder_teacher = TextEncoder(config)

        self.image_projection = ProjectionHead(config, embedding_dim=config.image_embedding)
        self.image_projection_student = ProjectionHead(config, embedding_dim=config.image_embedding)

        self.text_projection = ProjectionHead(config, embedding_dim=config.text_embedding)
        self.text_projection_student = ProjectionHead(config, embedding_dim=config.text_embedding)

        self.attend_vt = Dot_Attention(config.projection_size, config.projection_size, config.projection_size)
        self.attend_tv = Dot_Attention(config.projection_size, config.projection_size, config.projection_size)

        self.classifier = self.build_projection(config.projection_size * 4, 2)


    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
            nn.Softplus()
        )

    def build_projection(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, batch, epoch_number):
        #### task for uncertainty computation
        image_features = self.image_encoder(ids=batch['id'], image=batch["image"])
        text_features = self.text_encoder(ids=batch['id'],
                                          input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        evidence_text = self.text_classifier(F.normalize(text_features, p=2, dim=1))
        evidence_image = self.image_classifier(F.normalize(image_features, p=2, dim=1))
        evidence = {}
        evidence[0] = evidence_text
        evidence[1] = evidence_image

        belta = {}
        belta[0] = evidence[0] + 1.0 * self.scaler
        belta[1] = evidence[1] + 1.0 * self.scaler
        s = {}
        u = {}
        s[0] = torch.sum(belta[0], dim=1, keepdim=True)
        u[0] = 2.0 / s[0]
        s[1] = torch.sum(belta[1], dim=1, keepdim=True)
        u[1] = 2.0 / s[1]
        loss = 0
        alpha = dict()
        for v_num in range(2):
            alpha[v_num] = evidence[v_num] + 1.0 * self.scaler  
            loss += ce_loss(batch['label'], alpha[v_num], 2, epoch_number + 1, 50,
                            self.scaler)  
        alpha_a = DS_Combin(alpha, self.scaler)  
        evidence_a = alpha_a - 1.0 * self.scaler  
        loss += ce_loss(batch['label'], alpha_a, 2, epoch_number + 1, 50, self.scaler)  
        loss = torch.mean(loss)

        #####  task for fusion and  classifier
        image_features_teacher = self.image_encoder_teacher(ids=batch['id'], image=batch["image"])
        text_features_teacher = self.text_encoder_teacher(ids=batch['id'],
                                          input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        self.image_embeddings = self.image_projection(image_features_teacher)

        self.text_embeddings = self.text_projection(text_features_teacher)

        att_v = self.attend_vt(self.image_embeddings, self.text_embeddings)
        att_t = self.attend_tv(self.text_embeddings, self.image_embeddings)

        multi_embedding = torch.cat((att_v, att_t), dim=1)

        self.multimodal_embeddings = torch.cat([self.image_embeddings * (1 - u[1]),
                                                self.text_embeddings *(1 - u[0]) ,
                                                multi_embedding * (u[0] + u[1])
                                                ], dim=1)
        score = self.classifier(self.multimodal_embeddings)

        score = F.softmax(score, dim=1)

        c_loss = self.classifier_loss_function(score, batch['label'])

        return score, evidence_a, loss , c_loss + loss


def calculate_loss(model):
    # 处理DDP包装的模型
    from torch.nn.parallel import DistributedDataParallel as DDP
    model_params = model.module if isinstance(model, DDP) else model
    
    s_loss = calculate_similarity_loss(model_params.image_embeddings, model_params.text_embeddings)
    c_loss = 0.0
    loss = s_loss
    return loss, c_loss, s_loss


def calculate_similarity_loss(image_embeddings, text_embeddings):
    logits = (text_embeddings @ image_embeddings.T)
    images_similarity = (image_embeddings @ image_embeddings.T)
    texts_similarity = (text_embeddings @ text_embeddings.T)
    targets = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
    texts_loss = cross_entropy(logits, targets, reduction='mean')
    images_loss = cross_entropy(logits.T, targets.T, reduction='mean')
    loss = (images_loss + texts_loss) / 2.0
    return loss


def cross_entropy(preds, targets, reduction='none'):
    # entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
    # return entropy(preds, targets)
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, scaler):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1.0 * scaler
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1.0 * scaler
    B = annealing_coef * KL(alp, c)

    return (A + B)


def DS_Combin(alpha, scaler):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):  # view要改
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)  # S = sigma(alpha)
            E[v] = alpha[v] - 1.0 * scaler  # E代表evidence, E = alpha -1
            b[v] = E[v] / (S[v].expand(E[v].shape))  # b = E / S
            u[v] = 2.0 * scaler / S[v]  # u = K /S

        # Definition 3.1 (Dempster’s combination rule for two independent sets of masses)
        # 构造矩阵 求 M1 + M2
        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, 2, 1), b[1].view(-1, 1, 2))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # 利用矩阵计算新的b, u, s ,e
        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))

        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = 2.0 * scaler / u_a

        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))

        alpha_a = e_a + 1.0 * scaler
        return alpha_a

    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
    return alpha_a


def _get_positive_mask(batch_size):
    diag = np.eye(batch_size)
    mask = torch.from_numpy((diag))
    mask = (1 - mask)
    return mask.cuda(non_blocking=True)


def compute_loss(logits, mask):
    return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))


def contrast(video_features, text_features, temperature, negative_w):
    """
    Inputs shape (batch, embed_dim)
    Args:
        im: Visual embeddings (batch, embed_dim)
        s: Text embeddings (batch, embed_dim)
    Returns:
    """
    batch_size = video_features.shape[0]

    # Normalize features
    video_features = nn.functional.normalize(video_features, dim=1)
    text_features = nn.functional.normalize(text_features, dim=1)

    # Inter-modality alignment
    logits_per_vid = video_features @ text_features.t()
    logits_per_text = text_features @ video_features.t()

    # Intra-modality alignment
    logits_clstr_vid = video_features @ video_features.t()
    logits_clstr_txt = text_features @ text_features.t()

    logits_per_vid /= temperature
    logits_per_text /= temperature
    logits_clstr_vid /= temperature
    logits_clstr_txt /= temperature

    positive_mask = _get_positive_mask(video_features.shape[0])
    negatives_vid = logits_clstr_vid * positive_mask
    negatives_txt = logits_clstr_txt * positive_mask

    vid_logits = torch.cat([logits_per_vid, negative_w * negatives_vid], dim=1)
    txt_logits = torch.cat([logits_per_text, negative_w * negatives_txt], dim=1)

    diag = np.eye(batch_size)
    mask_vid = torch.from_numpy((diag)).cuda()
    mask_txt = torch.from_numpy((diag)).cuda()

    mask_neg_v = torch.zeros_like(negatives_vid)
    mask_neg_t = torch.zeros_like(negatives_txt)
    mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
    mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

    loss_i = compute_loss(vid_logits, mask_v)
    loss_t = compute_loss(txt_logits, mask_t)

    return ((loss_i.mean() + loss_t.mean())) / 2

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs

        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def attention(Q, K, V):
    # 计算注意力分数
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))

    # 对注意力分数进行 softmax 操作
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 计算加权和
    output = torch.matmul(attention_weights, V)

    return output


def get_multi_modal_emb(mv, mt):
    """
    更新单模态嵌入向量
    参数:
        mv (torch.Tensor): 输入mv的张量
        mt (torch.Tensor): 输入mt的张量
    返回:
        torch.Tensor: 更新后的单模态嵌入向量
    """
    ftv = F.softmax(torch.matmul(mv, mt.transpose(0, 1)) / 8.0, dim=1)
    fvt = F.softmax(torch.matmul(mt, mv.transpose(0, 1)) / 8.0, dim=1)

    mvf = ftv @ mt
    mtf = fvt @ mv

    multi = torch.cat([mvf, mtf], dim=1)
    return multi
