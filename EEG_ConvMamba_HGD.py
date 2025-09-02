import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, precision_score, recall_score, \
    f1_score

# from Confusion_matrix import plot_confusion_matrix

# from sklearn.metrics import plot_confusion_matrix
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import datetime
import time
import scipy
import random
import numpy as np
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple

from torch import Tensor
from typing import Optional
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from rope import *
from timm.models.layers import trunc_normal_, lecun_normal_
from einops.layers.torch import Rearrange, Reduce
from sklearn import metrics
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super(Block, self).__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"

            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self,
                hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_devide_out=if_devide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        dim=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block


class VisionMamba(nn.Module):
    def __init__(self,
                 depth=4,
                 embed_dim=192,
                 num_classes=4,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 final_pool_type='all',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}

        kwargs.update(factory_kwargs)
        super(VisionMamba, self).__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim


        num_patches = self.embed_dim  # (26, 40)

        # cls_token
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # position embedding
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)


        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

    def forward_features(self, x, inference_params=None,
                         if_random_cls_token_position=False,
                         if_random_token_rank=False):

        B, M, _ = x.shape
        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)

                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]

            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value", x[0, token_position, 0])
            print("original token_position: ", token_position)

            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
                                      range(len(token_position))]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layer[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual == None else residual.flip([1]),
                    inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                residual_in_fp32=self.residual_in_fp32)

        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x,
                return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        Mamba_output = self.forward_features(x, inference_params,
                                   if_random_cls_token_position=if_random_cls_token_position,
                                   if_random_token_rank=if_random_token_rank)
        return Mamba_output


class Classificatin_FC(nn.Sequential):
    def __init__(self, n_classes=4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Feature_extract(nn.Module):
    def __init__(self, chans=44, samples=1000, dropout_rate=0.5, F1=10, F2=40):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, 15)),
                                   nn.BatchNorm2d(F1))
        self.conv2 = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, 15)),
                                   nn.BatchNorm2d(F1))
        self.conv3 = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, 15)),
                                   nn.BatchNorm2d(F1))
        self.conv4 = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, 15)),
                                   nn.BatchNorm2d(F1))

        self.Spatial_features = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(chans, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 75), (1, 12)),
            nn.Dropout(dropout_rate)
        )

        self.projection = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 1), stride=(1, 1)),  # 1×1 conv could enhance fiting ability slightly
            nn.AvgPool2d((1, 3)),
            nn.Dropout(dropout_rate),
            Rearrange('b e (h) (w) -> b (h w) e'),  # Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, x):
        # Temporal conv
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        # Concat data
        concat_feature = torch.cat((x1, x2, x3, x4), dim=1)
        # Spatial conv
        feature = self.Spatial_features(concat_feature)
        # enhance fiting ability slightly + sampling
        output = self.projection(feature)
        return output



class EEG_Mamba(nn.Sequential):
    def __init__(self, emb_size=40, depth=4, n_classes=4, **kwargs):  # 40
        super().__init__(
            Feature_extract(
                chans=22,
                samples=1000,
                dropout_rate=0.5,
                F1=10,
                F2=40
            ),

            VisionMamba(
                embed_dim=40,
                depth=depth,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                final_pool_type="all",
                if_abs_pos_embed=False,
                if_rope=False,
                if_rope_residual=False,
                bimamba_type="V2",
                if_cls_token=False,
                if_devide_out=True,
                use_middle_cls_token=True
            ),
            Classificatin_FC( n_classes=4)
        )


class Start():
    def __init__(self, nsub):
        super(Start, self).__init__()
        if nsub == 1:
            self.batch_size = 160
        if nsub != 1:
            self.batch_size = 220
        self.n_epochs = 2000
        # self.c_dim = 4
        self.lr = 0.0005
        self.b1 = 0.9
        self.b2 = 0.999
        self.nSub = 4
        self.start_epoch = 0
        self.root1 = '/Dataset/train/'
        self.root2 = '/Dataset/test/'
        self.log_write = open("/Dataset/results/log_subject%d.txt" % self.nSub, "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = EEG_Mamba().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))


    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug )
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 44, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(10):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 10)
                    tmp_aug_data[ri, :, :, rj * 100:(rj + 1) * 100] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 100:(rj + 1) * 100]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label



    def get_source_data(self):

        train_data = np.load(self.root1 + 's%d_data.npy' % self.nSub)
        train_label = np.load(self.root1 + 's%d_label.npy' % self.nSub)
        test_data = np.load(self.root2 + 's%d_data.npy' % self.nSub)
        test_label = np.load(self.root2 + 's%d_label.npy' % self.nSub)
        train_data = np.expand_dims(train_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        # standardize
        target_mean = np.mean(train_data)
        target_std = np.std(train_data)
        self.allData = (train_data - target_mean) / target_std
        self.testData = (test_data - target_mean) / target_std

        self.allLabel = train_label
        self.testLabel = test_label

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                           shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2),
                                          weight_decay=0.05)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        averAcc = 0
        averF1 = 0
        averPrecision = 0
        averRecall = 0
        num = 0
        bestAcc = 0
        bestF1 = 0
        bestPrecision = 0
        bestRecall = 0
        averKappa = 0

        # Train model
        for e in range(self.n_epochs):

            # Train model
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))


                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # test process
                if (e + 1) % 1 == 0:
                    self.model.eval()
                    Tok, Cls = self.model(test_data)

                    loss_test = self.criterion_cls(Cls, test_label)
                    y_pred = torch.max(Cls, 1)[1]

                    # Convert tensors to numpy arrays for sklearn metrics
                    test_label_np = test_label.cpu().numpy()
                    y_pred_np = y_pred.cpu().numpy()
                    label_np = label.cpu().numpy()
                    train_pred_np = torch.max(outputs, 1)[1].cpu().numpy()

                    # Calculate metrics
                    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                    train_acc = float((train_pred_np == label_np).astype(int).sum()) / float(label.size(0))

                    # Test metrics
                    test_precision = precision_score(test_label_np, y_pred_np, average='weighted')
                    test_recall = recall_score(test_label_np, y_pred_np, average='weighted')
                    test_f1 = f1_score(test_label_np, y_pred_np, average='weighted')
                    test_kappa = cohen_kappa_score(test_label_np, y_pred_np)

                    # Train metrics
                    train_precision = precision_score(label_np, train_pred_np, average='weighted')
                    train_recall = recall_score(label_np, train_pred_np, average='weighted')
                    train_f1 = f1_score(label_np, train_pred_np, average='weighted')
                    train_kappa = cohen_kappa_score(label_np, train_pred_np)

                    print('Epoch:', e,
                          '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                          '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                          '\n  Train accuracy: %.6f' % train_acc,
                          '  Train precision: %.6f' % train_precision,
                          '  Train recall: %.6f' % train_recall,
                          '  Train F1: %.6f' % train_f1,
                          '  Train Kappa: %.6f' % train_kappa,
                          '\n  Test accuracy: %.6f' % acc,
                          '  Test precision: %.6f' % test_precision,
                          '  Test recall: %.6f' % test_recall,
                          '  Test F1: %.6f' % test_f1,
                          '  Test Kappa: %.6f' % test_kappa)

                    # Update metrics
                    num += 1
                    averAcc += acc
                    averF1 += test_f1
                    averPrecision += test_precision
                    averRecall += test_recall
                    averKappa += test_kappa

                    if acc > bestAcc:
                        bestAcc = acc
                        bestF1 = test_f1
                        bestPrecision = test_precision
                        bestRecall = test_recall
                        bestKappa = test_kappa
                        Y_true = test_label
                        Y_pred = y_pred

            # Calculate averages
            averAcc /= num
            averF1 /= num
            averPrecision /= num
            averRecall /= num
            averKappa /= num

            # Print final metrics
            print('\nFinal Metrics:')
            print('The average accuracy is: %.6f' % averAcc)
            print('The best accuracy is: %.6f' % bestAcc)
            print('The average precision is: %.6f' % averPrecision)
            print('The best precision is: %.6f' % bestPrecision)
            print('The average recall is: %.6f' % averRecall)
            print('The best recall is: %.6f' % bestRecall)
            print('The average F1 score is: %.6f' % averF1)
            print('The best F1 score is: %.6f' % bestF1)
            print('The average Kappa score is: %.6f' % averKappa)  # 添加Kappa输出
            print('The best Kappa score is: %.6f' % bestKappa)  # 添加Kappa输出

            # Write to log file
            self.log_write.write(f'The average accuracy is: {averAcc:.6f}\n')
            self.log_write.write(f'The best accuracy is: {bestAcc:.6f}\n')
            self.log_write.write(f'The average precision is: {averPrecision:.6f}\n')
            self.log_write.write(f'The best precision is: {bestPrecision:.6f}\n')
            self.log_write.write(f'The average recall is: {averRecall:.6f}\n')
            self.log_write.write(f'The best recall is: {bestRecall:.6f}\n')
            self.log_write.write(f'The average F1 score is: {averF1:.6f}\n')
            self.log_write.write(f'The best F1 score is: {bestF1:.6f}\n')
            self.log_write.write(f'The average Kappa score is: {averKappa:.6f}\n')  # 添加Kappa记录
            self.log_write.write(f'The best Kappa score is: {bestKappa:.6f}\n')  # 添加Kappa记录

            return (bestAcc, averAcc, bestPrecision, averPrecision,
                    bestRecall, averRecall, bestF1, averF1, bestKappa, averKappa,
                    Y_true, Y_pred)

def main():

    total_best_acc = 0
    total_aver_acc = 0
    total_best_pre = 0
    total_aver_pre = 0
    total_best_re = 0
    total_aver_re = 0
    total_best_f1 = 0
    total_aver_f1 = 0
    total_best_kappa = 0
    total_aver_kappa = 0
    num_subjects = 14

    result_write = open("/Dataset/results/sub_result.txt", "w")

    for i in range(14):
        starttime = datetime.datetime.now()

        seed_n = random.randint(1, 2025)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        exp = Start(i + 1)


        (bestAcc, averAcc, bestpre, averpre, bestre, averre,
         bestf1, averf1, bestkappa, averkappa, Y_true, Y_pred) = exp.train()

        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best Precision is: ' + str(bestpre) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average Precision is: ' + str(averpre) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best Recall is: ' + str(bestre) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average Recall is: ' + str(averre) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best F1-score is: ' + str(bestf1) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average F1-score is: ' + str(averf1) + "\n")
        result_write.write(
            'Subject ' + str(i + 1) + ' : ' + 'The best Kappa is: ' + str(bestkappa) + "\n")
        result_write.write(
            'Subject ' + str(i + 1) + ' : ' + 'The average Kappa is: ' + str(averkappa) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))


        total_best_acc += bestAcc
        total_aver_acc += averAcc
        total_best_pre += bestpre
        total_aver_pre += averpre
        total_best_re += bestre
        total_aver_re += averre
        total_best_f1 += bestf1
        total_aver_f1 += averf1
        total_best_kappa += bestkappa
        total_aver_kappa += averkappa

        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    avg_best_acc = total_best_acc / num_subjects
    avg_aver_acc = total_aver_acc / num_subjects
    avg_best_pre = total_best_pre / num_subjects
    avg_aver_pre = total_aver_pre / num_subjects
    avg_best_re = total_best_re / num_subjects
    avg_aver_re = total_aver_re / num_subjects
    avg_best_f1 = total_best_f1 / num_subjects
    avg_aver_f1 = total_aver_f1 / num_subjects
    avg_best_kappa = total_best_kappa / num_subjects
    avg_aver_kappa = total_aver_kappa / num_subjects

    # 写入最终平均结果
    result_write.write('\n**Final Average Results Across All Subjects:**\n')
    result_write.write(f'The average Best accuracy is: {avg_best_acc:.4f}\n')
    result_write.write(f'The average Average accuracy is: {avg_aver_acc:.4f}\n')
    result_write.write(f'The average Best Precision is: {avg_best_pre:.4f}\n')
    result_write.write(f'The average Average Precision is: {avg_aver_pre:.4f}\n')
    result_write.write(f'The average Best Recall is: {avg_best_re:.4f}\n')
    result_write.write(f'The average Average Recall is: {avg_aver_re:.4f}\n')
    result_write.write(f'The average Best F1-score is: {avg_best_f1:.4f}\n')
    result_write.write(f'The average Average F1-score is: {avg_aver_f1:.4f}\n')
    result_write.write(f'The average Best Kappa is: {avg_best_kappa:.4f}\n')
    result_write.write(f'The average Average Kappa is: {avg_aver_kappa:.4f}\n')

    yt_np = yt.cpu().numpy()
    yp_np = yp.cpu().numpy()

    result_write.write('\n**Overall Classification Report:**\n')
    clf_report = classification_report(yt_np, yp_np, digits=4)
    result_write.write(clf_report + '\n')

    result_write.write('\n**Confusion Matrix:**\n')
    conf_mat = confusion_matrix(yt_np, yp_np)
    result_write.write(np.array2string(conf_mat, separator=', ') + '\n')

    overall_kappa = cohen_kappa_score(yt_np, yp_np)
    result_write.write(f'\nOverall Kappa Score: {overall_kappa:.4f}\n')

    result_write.close()


    print('\n**Final Average Results Across All Subjects:**')
    print(f'Average Best Accuracy: {avg_best_acc:.4f}')
    print(f'Average Average Accuracy: {avg_aver_acc:.4f}')
    print(f'Average Best Precision: {avg_best_pre:.4f}')
    print(f'Average Average Precision: {avg_aver_pre:.4f}')
    print(f'Average Best Recall: {avg_best_re:.4f}')
    print(f'Average Average Recall: {avg_aver_re:.4f}')
    print(f'Average Best F1-score: {avg_best_f1:.4f}')
    print(f'Average Average F1-score: {avg_aver_f1:.4f}')
    print(f'Average Best Kappa: {avg_best_kappa:.4f}')
    print(f'Average Average Kappa: {avg_aver_kappa:.4f}')
    print(f'\nOverall Kappa Score: {overall_kappa:.4f}')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
