import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
import dgl
import copy
import heapq
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import random
from collections import Counter
from pygod.metric import * 
import re
from torch_geometric.utils import remove_isolated_nodes


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()  #, sparse_to_tuple(features)
 

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_matrix(adj)    
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)   #target节点放在最后一个位置

    return subv


def get_scores(actual,score,k): 
    auc = eval_roc_auc(actual,score) 
    ap = eval_average_precision(actual,score) 
    rec = eval_recall_at_k(actual,score,k) 
    return auc, ap, rec


def get_one_sample(subgraph_size,nb_nodes,coef,subgraphs,strategy):
    all_samples = []
    for nd in range(nb_nodes):
        nes = list(set(subgraphs[nd]))
        nes.remove(nd)
        if nes == []:
            nes = [nd]
        
        if strategy =='random':
            one_sample =  random.sample(nes,1)
            sample_id = subgraphs[nd].index(one_sample[0])
            all_samples.append(sample_id)
    
            
        elif strategy == 'most-relevant':
            coefs= [coef[nd][neb]  for neb in nes]
            max_index = map(coefs.index, heapq.nlargest(1, coefs)) 
            maxs = list(set(max_index))
            maxid =  subgraphs[nd].index(nes[maxs[0]]) 
            all_samples.append(maxid)
        
        elif strategy == 'least-relevant':
            coefs= [coef[nd][neb]  for neb in nes]
            min_index = map(coefs.index, heapq.nsmallest(1, coefs)) 
            mins = list(set(min_index))
            minid = subgraphs[nd].index(nes[mins[0]])  
            all_samples.append(minid)
            
    all_samples = np.array(all_samples)
    return all_samples


def RemoveIsolated(data):
    num_nodes = data.num_nodes
    out = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes)
    data.edge_index, data.edge_attr, mask = out

    if hasattr(data, '__num_nodes__'):
        data.num_nodes = int(mask.sum())

    for key, item in data:
        if bool(re.search('edge', key)):
            continue
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            data[key] = item[mask]

    return data



    
def get_negs(idx,nodecom,communities,Com_size_ratio,num_negs,neg_sample_method):
    if neg_sample_method  == 'random':
        neg_node = [i for i in range(len(idx))]
        Negs = []
        for i in range(num_negs):
            each_negs =  random.choices(neg_node, k = len(idx)) 
            for i, value in enumerate(each_negs):
                if value == i:
                    if value!= max(neg_node):
                        each_negs[i] += 1
                    else:
                        each_negs[i] -= 1
            Negs.append(each_negs)
        multi_neg_node = np.array(Negs) 

    else:
        
        mapper =  {node: i for i, node in enumerate(idx)}
        Comnode = []
        for k in  range(len(communities)):
            nodes = [mapper[nd] for nd in idx if nodecom[nd]==k]
            Comnode.append(nodes)

        Negs = []
        for nd in idx:
            nd_com = nodecom[nd]
            if neg_sample_method == 'bias':
                nd_selected_com = random.choices(communities[:nd_com] + communities[nd_com+1:],weights = Com_size_ratio[nd_com], k = num_negs)
            if neg_sample_method  == 'even':
                nd_selected_com = random.choices(communities[:nd_com] + communities[nd_com+1:],weights = [1/(len(communities)-1)] * (len(communities)-1), k = num_negs)
            dic = Counter(nd_selected_com)
            neg_samples = [random.sample(Comnode[key],dic[key]) for key in dic]
            neg_samples = sum(neg_samples,[])
            Negs.append(neg_samples)
            multi_neg_node = np.array(Negs).T 
            
    return multi_neg_node






