import sys
import os

nb_dir = os.path.split(os.getcwd())[0]
nb_dir = nb_dir.replace('conclusion_model_revamp', '')
sys.path.append(nb_dir)

os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
from conclusion_model_revamp.siamese_nn import sampling_techniques as sampler
from conclusion_model_revamp.conclusion_target_modeling import *

from networks import TargetEmbeddingNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss
import datasets

from conclusion_model_revamp.target_embedding_utils import *

batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
embedding_method='fasttext'

ranking_model = RankedTargetModel('../../data/idebate/ranking_model.pickle')

def grid_search_over_k(k_range, how_to_choose_k, test_file_path, output_path, test_scenario):
    
    target_space_path= '../target_spaces/fasttext_embedded_targets_space.pkl'
    target_space = pickle.load(open(target_space_path, 'rb'))

    train_data = json.load(open('../../data/idebate/train_scored.json', 'r'))
    val_data   = json.load(open('../../data/idebate/dev_scored.json', 'r'))
    test_data  = json.load(open(test_file_path, 'r'))

    with open(output_path, 'w') as f:
        f.write('k, empty_space, model acc @20, model acc @40, model acc @50, model bleu, model meteor, baseline acc @20, baseline acc @40, baseline acc @50, baseline bleu, baseline meteor\n')
        results = []
        for k in k_range:

           
            combination_of = k
            
            train_triplet_ds, train_p_avg, _ = prepare_data(train_data, target_space_file=target_space_path,
                                embedding_method=embedding_method, triple_sampling=True, top_k=k, how_to_choose_k=how_to_choose_k, combination_of=combination_of, ranking_model=ranking_model)
            val_triplet_ds, val_p_avg, _  = prepare_data(val_data, target_space_file=target_space_path,
                                embedding_method=embedding_method, triple_sampling=True, top_k=k, how_to_choose_k=how_to_choose_k, combination_of=combination_of, ranking_model=ranking_model)

            prepared_test_data = prepare_test_data(test_data, embedding_method, k, how_to_choose_k=how_to_choose_k, ranking_model=ranking_model)

            triplet_train_loader = torch.utils.data.DataLoader(train_triplet_ds, batch_size=batch_size, shuffle=True, **kwargs)
            triplet_test_loader = torch.utils.data.DataLoader(val_triplet_ds, batch_size=batch_size, shuffle=False, **kwargs)

            
            margin = 0.2
            target_embedding_net = TargetEmbeddingNet(300)
            #avg_embedding_net = TargetEmbeddingNet(300)
            triple_model = TripletNet(target_embedding_net, target_embedding_net)
            if cuda:
                triple_model.cuda()
            loss_fn = TripletLoss(margin)
            lr = 1e-4
            optimizer = optim.Adam(triple_model.parameters(), lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, 300, gamma=0.5, last_epoch=-1)
            n_epochs = 20
            log_interval = 100
            train_losses, val_losses, _ = fit(triplet_train_loader, triplet_test_loader, triple_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, callback_test=None)


            #Test the model on empty space
            empty_space_baseline_acc_at_thresholds, empty_space_model_acc_at_thresholds, \
            empty_space_baseline_meteor_score, empty_space_baseline_bleu_score, empty_space_model_meteor_score, empty_space_model_bleu_score, _, _ = test_model(triple_model, prepared_test_data, {}, 
                                                                                                            thresholds=[0.2, 0.4, 0.5], test_scenario=test_scenario,combination_of=combination_of)
            print('{}, {}, {}, {}'.format(k, 'yes', empty_space_model_acc_at_thresholds, empty_space_baseline_acc_at_thresholds))
            #Test the model on target space
            baseline_acc_at_thresholds, model_acc_at_thresholds,\
            baseline_meteor_score, baseline_bleu_score, model_meteor_score, model_bleu_score, _, _ = test_model(triple_model, prepared_test_data, target_space, 
                                                                                    thresholds=[0.2, 0.4, 0.5], test_scenario=test_scenario, combination_of=combination_of)

            print('{}, {}, {}, {}'.format(k, 'no', model_acc_at_thresholds, baseline_acc_at_thresholds))

            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(k, 'yes', *empty_space_model_acc_at_thresholds, empty_space_model_bleu_score, empty_space_model_meteor_score, 
                                                                                        *empty_space_baseline_acc_at_thresholds, empty_space_baseline_bleu_score, empty_space_baseline_meteor_score))
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(k, 'no', *model_acc_at_thresholds, model_bleu_score, model_meteor_score,
                                                                                       *baseline_acc_at_thresholds, baseline_bleu_score, baseline_meteor_score))


def tune_model_params(log_file_path):
    how_to_choose_k=False


    #target_space_path= '../target_spaces/glove_embedded_targets_space.pkl'
    target_space_path= '../target_spaces/fasttext_embedded_targets_space.pkl'
    #target_space_path= '../target_spaces/fasttext_embedded_cmv_targets_space.pkl'

    #cmv_train_data = json.load(open('../../data/cmv/train.json', 'r'))
    train_data = json.load(open('../../data/idebate/train_scored.json', 'r'))
    val_data   = json.load(open('../../data/idebate/dev_scored.json', 'r'))


    with open(log_file_path, 'w') as logfile:
        logfile.write('top_k, margin, epoch, acc@20, acc@40, acc@50, bleu, meteor\n')
        for top_k in [2, 4, 6, 8]:
            combination_of = top_k
            num_negs = 1

            train_triplet_ds, train_p_avg, new_target_space = prepare_data(train_data, target_space_file=target_space_path,
                                                               embedding_method=embedding_method, triple_sampling=True, 
                                                               top_k=top_k, how_to_choose_k=how_to_choose_k,
                                                            combination_of=combination_of, ranking_model=ranking_model, num_negs = num_negs)
            val_triplet_ds, val_p_avg, _  = prepare_data(val_data, target_space_file=target_space_path,
                                                    embedding_method=embedding_method, triple_sampling=True, top_k=top_k, 
                                                    how_to_choose_k=how_to_choose_k, combination_of=combination_of, ranking_model=ranking_model, num_negs = num_negs)



            triplet_train_loader = torch.utils.data.DataLoader(train_triplet_ds, batch_size=batch_size, shuffle=True, **kwargs)
            triplet_test_loader = torch.utils.data.DataLoader(val_triplet_ds, batch_size=batch_size, shuffle=False, **kwargs)

            # Prepare the test function
            target_space = pickle.load(open(target_space_path, 'rb'))
            test_data  = json.load(open('../../data/idebate/dev_scored.json', 'r'))
            test_data = prepare_test_data(test_data, embedding_method, top_k=top_k, how_to_choose_k=how_to_choose_k, ranking_model=ranking_model)
            test_fun = lambda model: test_model(model, test_data, target_space, test_scenario='optimistic',combination_of=combination_of)
            

            for margin in [0.1, 0.2, 0.5, 1.0]:
                print('Margin: ', margin)
                print('K:', top_k)
                target_embedding_net = TargetEmbeddingNet(300)
                #avg_embedding_net = TargetEmbeddingNet(300)
                triple_model = TripletNet(target_embedding_net, target_embedding_net)
                if cuda:
                    triple_model.cuda()
                loss_fn = TripletLoss(margin)
                lr = 1e-4
                optimizer = optim.Adam(triple_model.parameters(), lr=lr)
                scheduler = lr_scheduler.StepLR(optimizer, 300, gamma=0.5, last_epoch=-1)
                n_epochs = 31
                log_interval = 100
                train_losses, val_losses, metrices = fit(triplet_train_loader, triplet_test_loader, triple_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, callback_test=test_fun)

                for m in metrices:
                    logfile.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(top_k, margin, m['epoch'], *m['model_acc'], m['model_bleu'], m['model_meteor']))


#grid_search_over_k(k_range=range(1, 26), how_to_choose_k='order', test_file_path='../../data/idebate/dev_scored.json', output_path='./choosing_top_k_pessimistic.csv', test_scenario='pessimistic')
grid_search_over_k(k_range=range(1, 26), how_to_choose_k='order', test_file_path='../../data/idebate/dev_scored.json', output_path='./choosing_order_k_pessimistic.csv', test_scenario='pessimistic')

#tune_model_params('./tuning_appoach.csv')