import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            #保持张量维度不变
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)



class Model(nn.Module):
    def __init__(self, n_in, n_h, activation,readout, T ):
        super(Model, self).__init__()
        self.read_mode = readout
        self.T = T

        self.gcn_node = GCN(n_in, n_h, activation)
        self.gcn_context = GCN(n_in, n_h, activation)

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        
        
    def forward(self, bf_mask, ba, bf, ba_mask, multi_neg_node, sample_node, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
          
        h_1 = self.gcn_node(bf, ba_mask, sparse) 
        h_2 = self.gcn_context(bf_mask, ba, sparse)    
        
        target_node = h_1[:, -1, :]
        target_node = nn.functional.normalize(target_node, dim=1)
        
        #extract neighbor
        pos_individual_neighbor = h_1[[range(len(sample_node))],sample_node,:].squeeze()        
        pos_individual_neighbor = nn.functional.normalize(pos_individual_neighbor, dim=1)

        neg_individual_neighbor = torch.stack([h_1[node][[range(len(sample_node))],sample_node,:].squeeze() for node in multi_neg_node])              
        neg_individual_neighbor = nn.functional.normalize(neg_individual_neighbor, dim=2)
        
        #extract subgraph
        pos_sub = self.read(h_2[:, :-1, :])   
        neg_sub = torch.stack([pos_sub[node] for node in multi_neg_node])
         
        #target v.s. neighbor
        node_pos = torch.einsum('nc,nc->n', [target_node, pos_individual_neighbor]).unsqueeze(-1)
        # negative logits: NxK
        node_neg = torch.einsum('nc,knc->nk', [target_node, neg_individual_neighbor])
        # logits: Nx(1+K)
        node_logits = torch.cat([node_pos, node_neg], dim=1)
        
        #target v.s. subgraph
        sub_pos = torch.einsum('nc,nc->n', [target_node, pos_sub]).unsqueeze(-1)
        sub_neg = torch.einsum('nc,knc->nk', [target_node, neg_sub])
        sub_logits = torch.cat([sub_pos, sub_neg], dim=1)

        # apply temperature
        node_logits /= self.T
        sub_logits /= self.T

        labels = torch.zeros(node_logits.shape[0], dtype=torch.long)
        
        return node_logits, sub_logits, labels 


