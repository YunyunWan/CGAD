import os
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
import math
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import json
import pickle
from model import Model 
from utils import *
import torch_geometric.transforms as T 
import metis


def Rnce_loss(logits,lam,q):
    exps = torch.exp(logits)
    pos =  -(exps[:,0])**q/q
    neg = ((lam*(exps.sum(1)))**q) /q
    loss = pos.mean() + neg.mean()
    return loss


def train_ours(args):
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(args.runs)]

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    
                        ######################################################################
                        #                             load data                              #
                        ######################################################################

    data =  torch.load(args.dataset+'.pt')
    if data.has_isolated_nodes() == True:
        data = RemoveIsolated(data)

    adj = to_scipy_sparse_matrix(data.edge_index).tocsr() 
    features = sp.lil_matrix(np.array(data.x) )
    y = data.y.bool()    # binary labels (inlier/outlier)
    ano_label = np.array(y)
    

    features = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    adj = normalize_adj(adj)   
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    

    #load community information 
    with open( args.dataset+'.json',encoding = 'utf8') as f:
        line = f.readline()
        nc = json.loads(line)
    nodecom = nc['com']
    
    com_size = dict(Counter(nodecom))
    communities = sorted(com_size)

    Com_size_ratio = []
    for com in communities:
        other_sums = nb_nodes - com_size[com]
        seqs = [com_size[item]/other_sums for item in communities if item!= com ]
        Com_size_ratio.append(seqs)
                    
    comnode = []
    for item in communities:
        each_com_node = [nd for nd in range(len(nodecom)) if nodecom[nd] == item]
        comnode.append(each_com_node)

    #load coef of features
    fr = open( args.dataset + '.txt','rb')
    coef = pickle.load(fr)
    fr.close()


                                ######################################################################
                                #                             train model                            #
                                ######################################################################


    for run in range(args.runs):
        seed = seeds[run]
        print('---Train now---')       
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, args.embedding_dim, 'prelu',args.readout,args.T).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
      
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        batch_num = math.ceil(nb_nodes // batch_size) 

        for epoch in range(args.num_epoch): 
            
            model.train()
            total_loss = 0.


            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size) 
            all_subgraphs = np.array(subgraphs)                         
            all_samples = get_one_sample(subgraph_size,nb_nodes,coef,subgraphs,args.strategy)

            #com_batch_sizes: community size in each batch 
            com_batch_sizes = [len(item)//batch_num for item in comnode]
            for item in comnode:
                random.shuffle(item)


            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))        
                if not is_final_batch:
                    idx = [comnode[j][batch_idx*com_batch_sizes[j]:(batch_idx + 1)*com_batch_sizes[j]] for j in communities]
                    idx = sum(idx,[])
                else:
                    idx = [comnode[j][batch_idx*com_batch_sizes[j]:] for j in communities]
                    idx = sum(idx,[])
                    
                cur_batch_size = len(idx)
 
            
                  ###########################################################
                  #                  sample negative node                   #
                  ###########################################################

                random.shuffle(idx)
                multi_neg_node = get_negs(idx,nodecom,communities,Com_size_ratio,args.num_negs,args.neg_sample_method)


                  ###########################################################
                  #            construct ba and bf of subgrpah              #
                  ###########################################################
                  

                ba = []
                bf = []

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    
                    ba.append(cur_adj)
                    bf.append(cur_feat)                    
                
                ba = torch.cat(ba)
                bf = torch.cat(bf)

                #mask features
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                bf_mask = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1).to(device)    
                
                #mask structures
                mask_adj = torch.eye(subgraph_size).to(device)
                ba_mask = mask_adj.expand(cur_batch_size, subgraph_size, subgraph_size).to(device)
                
                
                sample_node = all_samples[idx]
                node_logits, sub_logits, labels  = model(bf_mask, ba, bf, ba_mask,multi_neg_node, sample_node)


                if args.loss_fun == 'rnce':
                    node_loss = Rnce_loss(node_logits,lam = args.lam, q = args.q)
                    sub_loss = Rnce_loss(sub_logits,lam = args.lam, q = args.q)
                    
            
                    
                #loss 
                loss =  args.alpha* node_loss +  (1 - args.alpha)*sub_loss
                loss.backward()
                optimiser.step()
                
                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
            
            
            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1
            
            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
            
            

                            ######################################################################
                            #             test model, compute anomaly score                      #
                            ######################################################################
       
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('---Test now---')
        for round in range(args.auc_test_rounds):
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            all_subgraphs = np.array(subgraphs)                     
            all_samples = get_one_sample(subgraph_size,nb_nodes,coef,subgraphs,args.strategy)
            com_batch_sizes = [len(item)//batch_num for item in comnode]

            for item in comnode:
                random.shuffle(item)
                
            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))        
                if not is_final_batch:
                    idx = [comnode[j][batch_idx*com_batch_sizes[j]:(batch_idx + 1)*com_batch_sizes[j]] for j in communities]
                    idx = sum(idx,[])
                else:
                    idx = [comnode[j][batch_idx*com_batch_sizes[j]:] for j in communities]
                    idx = sum(idx,[])
                    
                cur_batch_size = len(idx)


                  ###########################################################
                  #                  sample negative node                   #
                  ###########################################################
                
                multi_neg_node = get_negs(idx,nodecom,communities,Com_size_ratio,args.num_negs,args.neg_sample_method)


              ###########################################################
              #          construct adj and fea of subgrpah              #
              ###########################################################
                ba = []
                bf = []

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                bf = torch.cat(bf)
                
                #ba, bf_mask
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                bf_mask = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1).to(device)   
                
                #ba_mask, bf
                mask_adj = torch.eye(subgraph_size).to(device)
                ba_mask = mask_adj.expand(cur_batch_size, subgraph_size, subgraph_size).to(device)

                with torch.no_grad():
                    sample_node = all_samples[idx]
                    node_logits, sub_logits, labels  = model(bf_mask, ba, bf, ba_mask,multi_neg_node, sample_node)
                
                node_score = node_logits[:, 1:].detach().numpy() - node_logits[:, 0].unsqueeze(1).detach().numpy()
                node_score = np.mean(node_score,1) + np.std(node_score, 1)

                sub_score = sub_logits[:, 1:].detach().numpy() - sub_logits[:, 0].unsqueeze(1).detach().numpy()
                sub_score = np.mean(sub_score,1) + np.std(sub_score, 1)
                
                ano_score = args.alpha * node_score +  (1 - args.alpha) *sub_score 
                multi_round_ano_score[round, idx] = ano_score
    
        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        ano_score_final = ano_score_final - min(ano_score_final)  #keep>0
 
    return ano_label,ano_score_final

