import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from utils import *

class TransE(torch.nn.Module):
    def __init__(self, params, ):
        super(TransE, self).__init__()

        self.p = params

        self.ent_embed = get_param((self.p.num_ent, self.p.embed_dim))
        self.rel_embed = get_param((self.p.num_rel*2, self.p.embed_dim))

        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label=None, sub_samp=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_n'):
        sub_emb = self.ent_embed[sub]
        rel_emb = self.rel_embed[rel]
        all_ent = self.ent_embed
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score

class DistMult(torch.nn.Module):
    def __init__(self, params, ):
        super(DistMult, self).__init__()

        self.p = params

        self.ent_embed = get_param((self.p.num_ent, self.p.embed_dim))
        self.rel_embed = get_param((self.p.num_rel*2, self.p.embed_dim))

        self.bceloss = torch.nn.BCELoss()

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.bn0 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.register_parameter(
            'bias', torch.nn.Parameter(torch.zeros(self.p.num_ent)))

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_n'):
        sub_emb = self.ent_embed[sub]
        rel_emb = self.rel_embed[rel]
        all_ent = self.ent_embed

        sub_emb = self.bn0(sub_emb)
        sub_emb = self.inp_drop(sub_emb)

        x = torch.mm(sub_emb * rel_emb, all_ent.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        pred = torch.sigmoid(x)

        return pred