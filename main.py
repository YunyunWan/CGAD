
import tqdm
import torch
import argparse
import warnings
from pygod.metrics import *
import os
import json
from run import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Com')
parser.add_argument('--expid', type=int)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dataset', type=str, default='inj_cora')
parser.add_argument('--readout', type=str, default='avg')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default = 0)
parser.add_argument('--embedding_dim', type=int, default = 64)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--num_epoch', type=int, default= 100)  
parser.add_argument('--batch_size', type=int, default = 256)
parser.add_argument('--subgraph_size', type=int, default = 4 )    
parser.add_argument('--auc_test_rounds', type=int, default = 100) 

parser.add_argument('--num_community', type=int, default = 3)   
parser.add_argument('--neg_sample_method', type=str, default = 'bias')  #bias,even, random 
parser.add_argument('--num_negs', type=int, default = 3)   

parser.add_argument('--strategy', type=str, default='most-relevant')  #random, least-relevant 
parser.add_argument('--alpha', type=float, default = 0.5, help='how much node-level involves')

parser.add_argument('--loss_fun', type=str, default = 'rnce', help='loss function')
parser.add_argument('--lam', type=float, default = 0.5, help ='how much neg pairs involves')
parser.add_argument('--T', type=float, default = 1)   
parser.add_argument('--q', type=float, default = 0.3)   

args = parser.parse_args()


lrs1= [1e-3, 5e-4]
lrs2 = [1e-2, 5e-3]
bs1 = [512, 1024]
bs2 = [32,64]
ems1 = [64,128]
ems2 = [12,16]
ems3 = [32,48]  


if args.dataset in ['inj_cora']:
    lrs = lrs1
    ems = ems1
    bs = bs1
    args.num_community = 10
    coms = [3,5,8,10,15]


if args.dataset in ['books']:
    lrs = lrs2
    ems = ems2
    bs = bs1
    args.num_community = 10
    coms = [3,5,8,10,15]
    

if args.dataset in [ 'disney']:
    lrs = lrs2
    ems = ems2
    bs = bs2   
    args.num_community = 3
#    num_negs = [1, 3, 5]
#    coms = [3, 4, 5, 8]

if args.dataset in ['reddit']:
    lrs = lrs1
    ems = ems3
    bs = bs1
    args.num_epoch = 10
    args.auc_test_rounds = 10
    args.num_community = 3
    coms = [3,5,8,10,15]



ave_results = []
for args.lr in lrs:
    for args.batch_size in bs:
        for args.embedding_dim in ems:
            print('\n==============================')
            print( args.lr, args.batch_size,args.embedding_dim)
        
            #train
            ano_label,ano_score_final = train_ours(args)
            k = sum(ano_label)
        
            #compute index
            auc, ap, recall = get_scores(ano_label,ano_score_final ,k)
            metric = [args.lr, args.batch_size,args.embedding_dim, auc, ap,recall]

            #average performance
            ave_results.append([auc, ap, recall])
            final_metric = np.mean(ave_results,0)
            
            






























