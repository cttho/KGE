import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from utils import *
from layers import *


class TransE(torch.nn.Module):
    def __init__(self, params, ):
        super(TransE, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(
            self.p.num_ent, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.ent_embed.weight)

        self.rel_embed = torch.nn.Embedding(
            self.p.num_rel, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.rel_embed.weight)

        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label=None, sub_samp=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_n'):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)
        all_ent = self.ent_embed
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score

class DistMult(torch.nn.Module):
    def __init__(self, params, ):
        super(DistMult, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(
            self.p.num_ent, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.ent_embed.weight)

        self.rel_embed = torch.nn.Embedding(
            self.p.num_rel*2, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.rel_embed.weight)

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

    def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)
        sub_emb = self.bn0(sub_emb)
        sub_emb = self.inp_drop(sub_emb)

        if strategy == 'one_to_n':
            x = torch.mm(sub_emb * rel_emb,
                         self.ent_embed.weight.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            x = torch.mul((sub_emb * rel_emb).unsqueeze(1),
                          self.ent_embed(neg_ents)).sum(dim=-1)
            x += self.bias[neg_ents]

        pred = torch.sigmoid(x)

        return pred


class ComplEx(torch.nn.Module):
    def __init__(self, params, ):
        super(ComplEx, self).__init__()

        self.p = params

        self.ent_embed_real = torch.nn.Embedding(
            self.p.num_ent, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.ent_embed_real.weight)

        self.ent_embed_imaginary = torch.nn.Embedding(
            self.p.num_ent, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.ent_embed_imaginary.weight)

        self.rel_embed_real = torch.nn.Embedding(
            self.p.num_rel*2, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.rel_embed_real.weight)

        self.rel_embed_imaginary = torch.nn.Embedding(
            self.p.num_rel*2, self.p.embed_dim, padding_idx=None)
        torch.nn.init.xavier_normal_(self.rel_embed_imaginary.weight)

        self.bceloss = torch.nn.BCELoss()

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.bn0 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.register_parameter(
            'bias', torch.nn.Parameter(torch.zeros(self.p.num_ent)))

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
        sub_emb_real = self.ent_embed_real(sub)
        sub_emb_imaginary = self.ent_embed_imaginary(sub)

        rel_emb_real = self.rel_embed_real(rel)
        rel_emb_imaginary = self.rel_embed_imaginary(rel)

        sub_emb_real = self.bn0(sub_emb_real)
        sub_emb_real = self.inp_drop(sub_emb_real)

        sub_emb_imaginary = self.bn0(sub_emb_imaginary)
        sub_emb_imaginary = self.inp_drop(sub_emb_imaginary)

        if strategy == 'one_to_n':
            x = torch.mm(sub_emb_real*rel_emb_real, self.ent_embed_real.weight.transpose(1, 0)) +\
                torch.mm(sub_emb_real*rel_emb_imaginary, self.ent_embed_imaginary.weight.transpose(1, 0)) +\
                torch.mm(sub_emb_imaginary*rel_emb_real, self.ent_embed_imaginary.weight.transpose(1, 0)) -\
                torch.mm(sub_emb_imaginary*rel_emb_imaginary,
                         self.ent_embed_real.weight.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            neg_embs_real = self.ent_embed_real(neg_ents)
            neg_embs_imaginary = self.ent_embed_imaginary(neg_ents)

            x = (torch.mul((sub_emb_real*rel_emb_real).unsqueeze(1), neg_embs_real) +
                 torch.mul((sub_emb_real*rel_emb_imaginary).unsqueeze(1), neg_embs_imaginary) +
                 torch.mul((sub_emb_imaginary*rel_emb_real).unsqueeze(1), neg_embs_imaginary) -
                 torch.mul((sub_emb_imaginary*rel_emb_imaginary).unsqueeze(1), neg_embs_real)).sum(dim=-1)

            x += self.bias[neg_ents]

        pred = torch.sigmoid(x)

        return pred