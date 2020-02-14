import sys
import os

nb_dir = os.path.split(os.getcwd())[0]
nb_dir = nb_dir.replace('conclusion_model_revamp', '')
sys.path.append(nb_dir)


import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
from scipy.spatial import distance

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
import random
import argparse
import pandas as pd
import pickle

from conclusion_model_revamp.ranking_targets import *
from conclusion_model_revamp import utils as utils
from conclusion_model_revamp.conclusion_target_modeling import *
from datasets import *
from target_embedding_utils import *
from networks import TargetEmbeddingNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss


batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

embedding_method  ='fasttext'
#target_space_path = '../target_spaces/fasttext_embedded_targets_space.pkl'

def load_data_to_df(train_data, val_data, test_samples, target_space, ranking_model, top_k, combination_of, num_negs):
    train_ds = prepare_data_as_text(train_data, target_space, top_k, 'rank', combination_of, num_negs, 'fasttext', ranking_model)
    valid_ds = prepare_data_as_text(val_data, target_space, top_k, 'rank', combination_of, num_negs, 'fasttext', ranking_model)

    train_df = pd.DataFrame(train_ds, columns=['p_targets', 'true_conc_target', 'false_conc_target', 
        'p_vectors', 'p_vectors_avg', 'conc_vector', 'other_conc_vector'])
    valid_df = pd.DataFrame(valid_ds, columns=['p_targets', 'true_conc_target', 'false_conc_target', 
        'p_vectors', 'p_vectors_avg', 'conc_vector', 'other_conc_vector'])
    

    test_data = prepare_test_data(test_samples, embedding_method, top_k=top_k, how_to_choose_k='rank', 
                                  ranking_model=ranking_model)

    return train_df, valid_df, test_data

def load_data(train_data, val_data, test_samples, target_space_path, ranking_model, top_k, combination_of, num_negs):

    train_triplet_samples, train_p_avg, new_target_space = prepare_data(train_data, target_space_file=target_space_path, embedding_method=embedding_method, 
                                                                        triple_sampling=True, top_k=top_k, how_to_choose_k='rank', combination_of=combination_of, 
                                                                        ranking_model=ranking_model, num_negs=num_negs)

    x0, x1, x2 = zip(*train_triplet_samples)
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)

    train_triplet_ds = TripletDataset(x0, x1, x2)

    val_triplet_samples, val_p_avg, _  = prepare_data(val_data, target_space_file=target_space_path, embedding_method=embedding_method, 
                                                        triple_sampling=True, top_k=top_k, how_to_choose_k='rank', combination_of=combination_of, 
                                                        ranking_model=ranking_model, num_negs=num_negs)

    x00, x11, x22 = zip(*val_triplet_samples)
    x00 = np.array(x00)
    x11 = np.array(x11)
    x22 = np.array(x22)

    val_triplet_ds = TripletDataset(x00, x11, x22)


    test_data = prepare_test_data(test_samples, embedding_method, top_k=top_k, how_to_choose_k='rank',  ranking_model=ranking_model)


    return train_triplet_ds, val_triplet_ds, test_data

def find_best_config(data_path, target_space_path, output_path, top_k):   
    ###### Loading data ##############
    ranking_model = RankedTargetModel(data_path + '/ranking_model.pickle')
    target_space  = pickle.load(open(target_space_path, 'rb'))
    train_data    = json.load(open(data_path + '/train_scored.json', 'r'))
    #train_data  = list(filter(lambda x: len(x['claims']) < 5, train_data))
    val_data      = json.load(open(data_path + '/dev_scored.json', 'r'))
    test_samples  = json.load(open(data_path + '/dev_scored.json', 'r'))

    best_configs=[]
    f = open(output_path + '/evaluation_{}.csv'.format(top_k), 'a+')
    f.write('k,combination_of,num_negs,margin,net_output,best_epoch,model_acc_@20,model_acc_@40,model_acc_@50,model_meteor,model_bleu,baseline_acc_@20,baseline_acc_@40,baseline_acc_@50,baseline_meteor,baseline_bleu\n')
    f.close()
    for num_negs in [1, 5]:
        for combination_of in range(1, top_k+1, 1):
            train_triplet_ds, val_triplet_ds, test_data = load_data(train_data, val_data, test_samples, target_space_path, ranking_model, top_k, combination_of, num_negs)
            #save the training and valdiation data
            #pickle.dump(train_triplet_ds, open(output_path + '/training_data_{}_{}_{}.pkl'.format(top_k, combination_of, num_negs), 'wb'))
            #pickle.dump(val_triplet_ds, open(output_path + '/valid_data_{}_{}_{}.pkl'.format(top_k, combination_of, num_negs), 'wb'))
            
            #train_triplet_ds = pickle.load(open(output_path + '/training_data_{}_{}_{}.pkl'.format(top_k, combination_of, num_negs), 'rb'))
            #val_triplet_ds  = pickle.load(open(output_path + '/valid_data_{}_{}_{}.pkl'.format(top_k, combination_of, num_negs), 'rb'))

            test_fun     = lambda model: test_model(model, test_data, target_space, combination_of=combination_of)
            triplet_train_loader = torch.utils.data.DataLoader(train_triplet_ds, batch_size=batch_size, shuffle=True, **kwargs)
            triplet_val_loader  = torch.utils.data.DataLoader(val_triplet_ds, batch_size=batch_size, shuffle=False, **kwargs)

            for margin in [0.2, 1.0]:
                for net_output in [300, 100]:
                    metrices = train_model(triplet_train_loader, triplet_val_loader, test_fun, margin, net_output,
                        model_directory=output_path + '/model_{}_{}_{}_{}_{}_'.format(top_k, combination_of, margin, net_output, num_negs))

                    sorted_metrices = sorted(metrices, key=lambda x: -x['model_bleu'])
                    best_configs.append({
                        'top_k': top_k,
                        'combination_of': combination_of,
                        'num_negs': num_negs,
                        'margin': margin,
                        'net_output': net_output,
                        'best_epoch': sorted_metrices[0]
                        })

                    print('BEST EPOCH:')
                    print(best_configs[-1])
                    print('===========================')
                    
                    f = open(output_path + '/evaluation_{}.csv'.format(top_k), 'a+')
                    f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(top_k,combination_of, num_negs, margin, net_output, *sorted_metrices[0].values()))
                    f.close()

    with open(output_path + '/evaluation_{}_{}.json'.format(top_k, num_negs), 'w') as f:
        json.dump(best_configs, f)

    return best_configs

def train_model(triplet_train_loader, triplet_val_loader, test_fun, margin, net_output, model_directory):

    target_embedding_net = TargetEmbeddingNet(300, net_output)
    triple_model = TripletNet(target_embedding_net, target_embedding_net)
    if cuda:
        triple_model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-4
    optimizer = optim.Adam(triple_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.5, last_epoch=-1)
    n_epochs = 20
    log_interval = 100

    train_losses, val_losses, metrices = fit(triplet_train_loader, triplet_val_loader, triple_model, 
                                             loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, 
                                             callback_test=test_fun, keep_checkpoint_max=10, model_dir=model_directory)


    return metrices



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--target_space_path', type=str)
    parser.add_argument('--top_k', type=int)

    args = parser.parse_args()

    find_best_config(args.data_path, args.target_space_path, args.output_path, args.top_k)