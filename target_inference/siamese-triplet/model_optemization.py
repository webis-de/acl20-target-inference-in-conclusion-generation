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

from conclusion_model_revamp.ranking_targets import *
from conclusion_model_revamp import utils as utils
from conclusion_model_revamp.conclusion_target_modeling import *
from datasets import *
from target_embedding_utils import *
from networks import TargetEmbeddingNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss

torch.manual_seed(7)
random.seed( 30 )

batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

embedding_method  ='fasttext'
target_space_path = '../target_spaces/fasttext_embedded_targets_space.pkl'

def find_best_model(top_k, combination_of, num_negs, how_to_choose_k, log_file):
    ###### Loading data ##############
    ranking_model = RankedTargetModel('../../data/idebate/ranking_model.pickle')

    
    train_data = json.load(open('../../data/idebate/train_scored.json', 'r'))
    val_data   = json.load(open('../../data/idebate/dev_scored.json', 'r'))

    train_triplet_samples, train_p_avg, new_target_space = prepare_data(train_data, target_space_file=target_space_path,
                                                                   embedding_method=embedding_method, triple_sampling=True, 
                                                                   top_k=top_k, how_to_choose_k=how_to_choose_k,
                                                                combination_of=combination_of, ranking_model=ranking_model, 
                                                                   num_negs=num_negs)

    x0, x1, x2 = zip(*train_triplet_samples)
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)

    train_triplet_ds = TripletDataset(x0, x1, x2)


    val_triplet_samples, val_p_avg, _  = prepare_data(val_data, target_space_file=target_space_path,
                                            embedding_method=embedding_method, triple_sampling=True, top_k=top_k, 
                                            how_to_choose_k=how_to_choose_k, combination_of=combination_of, ranking_model=ranking_model, 
                                             num_negs=num_negs)


    x00, x11, x22 = zip(*val_triplet_samples)
    x00 = np.array(x00)
    x11 = np.array(x11)
    x22 = np.array(x22)

    val_triplet_ds = TripletDataset(x00, x11, x22)



    triplet_train_loader = torch.utils.data.DataLoader(train_triplet_ds, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(val_triplet_ds, batch_size=batch_size, shuffle=False, **kwargs)

    # Prepare the test function
    target_space = pickle.load(open(target_space_path, 'rb'))
    test_data    = json.load(open('../../data/idebate/dev_scored.json', 'r'))
    test_data    = prepare_test_data(test_data, embedding_method, top_k=top_k, how_to_choose_k=how_to_choose_k, 
                                  ranking_model=ranking_model)
    test_fun     = lambda model: test_model(model, test_data, target_space)

    # Build the model
    margin = 0.2
    target_embedding_net = TargetEmbeddingNet(300)
    triple_model = TripletNet(target_embedding_net, target_embedding_net)
    if cuda:
        triple_model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-4
    optimizer = optim.Adam(triple_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.5, last_epoch=-1)
    n_epochs = 50
    log_interval = 100

    train_losses, val_losses, metrices = fit(triplet_train_loader, triplet_test_loader, triple_model, loss_fn, optimizer, scheduler,
                                                 n_epochs, cuda, log_interval, callback_test=test_fun)


    with open(log_file, 'w') as f:
        json.dump(metrices, f)

find_best_model(2, 2, 2, 'rank', './report_2_2_2_rank.json')