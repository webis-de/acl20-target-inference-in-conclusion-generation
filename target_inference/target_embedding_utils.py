import sys
import os
import itertools

nb_dir = os.path.split(os.getcwd())[0]
nb_dir = nb_dir.replace('target_inference', '')
sys.path.append(nb_dir)

os.environ["CUDA_VISIBLE_DEVICES"]="1"


import torch

import numpy as np
from scipy.spatial import distance

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
import random

from target_inference.target_ranking.ranking_targets import *
from target_inference import utils as utils
from target_inference import sampling_techniques as sampler

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.corpus import stopwords

class TargetsDataset(object):

    def __init__(self, p_targets, conc_targets, neg_targets):
        self.size = conc_targets.shape[0]
        self.p_targets    = torch.from_numpy(p_targets)
        self.conc_targets = torch.from_numpy(conc_targets)
        self.neg_targets  = torch.from_numpy(neg_targets)

    def __getitem__(self, index):
        return self.p_targets[index], self.conc_targets[index], self.neg_targets[index]

    def __len__(self):
        return self.size

def eval_meteor(reference, prediction):
    reference  = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), prediction))

    meteor_scores = [single_meteor_score(inst[0], inst[1]) for inst in zip(reference, prediction)]
    return round(sum(meteor_scores)/len(meteor_scores), 3)

def eval_bleu(reference, prediction, weights=[0.5, 0.5]):
    
    reference  = list(map(lambda x: [x.replace('<SCONC>', '').replace('<ECONC>', '').lower().split(' ')], reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower().split(' '), prediction))

    chencherry = SmoothingFunction()

    if weights != None:
        score = bleu_score.corpus_bleu(reference, prediction, weights=weights, smoothing_function=chencherry.method1)
    else:
        score = bleu_score.corpus_bleu(reference, prediction,  smoothing_function=chencherry.method1)

    return round(score * 100, 3)

def show_distance_dist(model_fn, ds, bins=np.arange(0, 2.1, 0.1), xticks=np.arange(0,2.1, 0.2)):
    from scipy.spatial import distance

    pre_pos_euc_sims = [distance.cosine(x[0][0].numpy(), x[0][1].numpy()) for x in ds]
    pre_neg_euc_sims = [distance.cosine(x[0][0].numpy(), x[0][2].numpy()) for x in ds]
    pre_pos_euc_sims = [2 if np.isnan(x) else x for x in pre_pos_euc_sims]
    pre_neg_euc_sims = [2 if np.isnan(x) else x for x in pre_neg_euc_sims]


    pos_euc_sims = [distance.cosine(model_fn(x[0][0]), 
                                    model_fn(x[0][1])) for x in ds]

    neg_euc_sims = [distance.cosine(model_fn(x[0][0]), 
                                    model_fn(x[0][2])) for x in ds]

    pos_euc_sims = [2 if np.isnan(x) else x for x in pos_euc_sims]
    neg_euc_sims = [2 if np.isnan(x) else x for x in neg_euc_sims]

    plt.hist(pos_euc_sims, alpha = 0.5, color='blue', label='mapped', bins=bins)
    plt.hist(pre_pos_euc_sims, alpha = 0.5, color='red', label='original',bins=bins)
    plt.title('distirbution of distance between pos and anchor')
    plt.xticks(xticks)
    plt.legend()
    plt.show()
    plt.hist(neg_euc_sims, alpha= 0.5, color='blue', label='mapped', bins=bins)
    plt.hist(pre_neg_euc_sims, alpha = 0.5, color='red', label='original',bins=bins)
    plt.title('distirbution of distance between neg and anchor')
    plt.xticks(xticks)
    plt.legend()
    plt.show()


def get_first_k_premise_targets(premises, k=None):
    all_premise_targets = [(target['text'], p['text']) for p in premises for target in p['targets']]
    
    return all_premise_targets[0:k]

def get_k_premise_targets(premises, k=None, choose_k_randomly=False, ranking_model=None):
    all_premise_targets = [(target['text'], p['text']) for p in premises for target in p['targets']]
    
    if k != None:
        if choose_k_randomly:
            #print('select random..')
            if k > len(all_premise_targets):
                return all_premise_targets
            else:
                return random.sample(all_premise_targets, k)
        else:
            top_k_targets = ranking_model.get_top_k_targets(premises, k)
            # print('top targets:')
            # print(top_k_targets)
            # print('================')
            return top_k_targets
    else:
        return all_premise_targets


def prepare_data(data, top_k=None, target_space_file=None, embedding_method=None, triple_sampling=False, how_to_choose_k='rank', combination_of=2, ranking_model=None, num_negs=1):
    samples = []
    premise_avg_pairs = []
    
    if target_space_file != None:
        target_space = pickle.load(open(target_space_file, 'rb'))
    else:
        target_space = {}

    new_target_space = {}
    for sample in data:
        conc = sample['conclusion']
        premises = sample['claims']
        conc_targets = conc['targets']
        conc_targets = (conc_targets[0]['text'], conc['text']) if len(conc_targets) > 0 else (conc['text'], conc['text'])
        
        if how_to_choose_k == 'order':
            all_premise_targets = get_first_k_premise_targets(premises, top_k)
        else:
            all_premise_targets = get_k_premise_targets(premises, top_k, choose_k_randomly= True if how_to_choose_k=='random' else False, ranking_model=ranking_model)

        if len(all_premise_targets) < 1: #if we only have one premise target to conclusion target
            continue

        if triple_sampling:
            c_samples , premise_avg_pair, negative_samples = sampler.triple_sampling(conc_targets, all_premise_targets, target_space, 
                                                                        embedding_method_name=embedding_method, combination_of=combination_of, num_neg_cases=num_negs)
            samples += c_samples
            premise_avg_pairs += premise_avg_pair
            
            for x in negative_samples.items():
                new_target_space[x[0]] = x[1]
        else:
            samples += sampler.technique_1(conc_targets, all_premise_targets, target_space, embedding_method_name=embedding_method)


    return samples, premise_avg_pairs, new_target_space

def prepare_test_data(test_data, embedding_method='glove', top_k=None,  how_to_choose_k='rank', ranking_model=None):
    data = []
    for sample in test_data:
        if len(sample['conclusion']['targets']) == 0:
            continue

        conc = sample['conclusion']['text']
        conc_target = sample['conclusion']['targets'][0]['text']
        
        if how_to_choose_k == 'order':
            top_p_targets = get_first_k_premise_targets(sample['claims'], top_k)
        else:
            top_p_targets = get_k_premise_targets(sample['claims'], top_k, choose_k_randomly= True if how_to_choose_k=='random' else False, ranking_model=ranking_model)

        all_p_targets = list(set([(target['text'], claim['text']) for claim in sample['claims'] for target in claim['targets']]))

        if len(all_p_targets) == 0:
            continue

        top_p_vectors   = [utils.embed_sentence(x[0], x[1], normalize=True, embedding_method_name= embedding_method) 
                           for x in top_p_targets]
        
        all_p_vectors   = [utils.embed_sentence(x[0], x[1], normalize=True, embedding_method_name= embedding_method) 
                           for x in all_p_targets]

        conc_target_vec = utils.embed_sentence(conc_target, conc, normalize=True, embedding_method_name= embedding_method)
        

        true_scores  = [compute_overlap(x[0], conc_target) for x in all_p_targets]
        
        all_p_targets, p_target_context = zip(*all_p_targets) #Take only the target phrase..
        
        data.append((all_p_vectors, top_p_vectors, true_scores, 
                     all_p_targets, top_p_targets, conc_target, conc_target_vec))

    return data

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def map_space(model, target_space):
    mapped_target_space = {}
    for chunk in chunks(list(target_space.items()), 10): #transform every 10 vectors at oncedistance_positive
        targets, vectors = zip(*chunk)
        vectors = np.array(vectors)
        #transform
        #vectors = (vectors + 1)/2
        tensor = torch.from_numpy(vectors)
        mapped_tensors = model.get_target_embedding(tensor).detach().numpy()

        for target, mapped_tensor in zip(targets, mapped_tensors):
            mapped_target_space[target] = mapped_tensor #/np.linalg.norm(mapped_tensor)

    print('Finshed Mapping...')
    return mapped_target_space

def test_model(model, test_data, target_space={}, thresholds=[0.2, 0.4, 0.5], test_scenario='optimistic', combination_of=2, input_texts=None):
    #map the target_space using the model..
    model.eval()
    with torch.no_grad():
        model_correct_cases_at_threshold = {threshold:0 for threshold in thresholds}
        baseline_correct_cases_at_threshold = {threshold:0 for threshold in thresholds}
        predicted_targets = []
        baseline_predicted_targets =[]
        true_targets = []
        
        if target_space != {}:
            lexicon_targets, lexicon_vectors = zip(*target_space.items())
            lexicon_targets, lexicon_vectors = list(lexicon_targets), list(lexicon_vectors)

            mapped_target_space = map_space(model, target_space)
            mapped_lexicon_targets, mapped_lexicon_vectors = zip(*mapped_target_space.items())
            mapped_lexicon_targets, mapped_lexicon_vectors = list(mapped_lexicon_targets), list(mapped_lexicon_vectors)
        else:
            lexicon_targets, lexicon_vectors = [], []
            mapped_lexicon_targets, mapped_lexicon_vectors = [], []
           
        idx = 0
        for (premise_targets_vectors, top_p_target_vectors,
             true_scores, premise_targets, top_p_targets, true_conc_target, true_conc_target_vec) in test_data:

            avg_vec     = np.mean(top_p_target_vectors, axis=0)
            premise_targets = list(premise_targets)
            
            #For the original space baseline...
            if test_scenario == 'optimistic':
                all_targets = premise_targets  + [true_conc_target] + lexicon_targets 
                all_vectors = premise_targets_vectors + [true_conc_target_vec] + lexicon_vectors
            else:
                all_targets = premise_targets + lexicon_targets 
                all_vectors = premise_targets_vectors + lexicon_vectors
            
            simple_avg_scores = [distance.cosine(avg_vec, x) for x in all_vectors]
            simple_avg_scores = [2 if np.isnan(x) else x for x in simple_avg_scores]
            simple_avg_pred = all_targets[np.argmin(simple_avg_scores)]
            
            #For the mapped/learned space approache..
            # 1. map premise targets into the new space
            mapped_premise_vectors = [model.get_target_embedding(torch.from_numpy(x)).detach().numpy() 
                                      for x in premise_targets_vectors]

            true_conc_mapped_vec = model.get_target_embedding(torch.from_numpy(true_conc_target_vec)).detach().numpy() 
            
            # 2. add the mapped premise targets into the whole lexicon of targets..
            if test_scenario == 'optimistic':
                all_mapped_vectors = mapped_premise_vectors + [true_conc_mapped_vec] + mapped_lexicon_vectors
            else:
                all_mapped_vectors = mapped_premise_vectors +  mapped_lexicon_vectors

            # 3. compute the average of top premise targets in the mapped space...
            
            #THE FOLLOWING ARE WAYS TO COMPUTE THE REPRESENTATIVE OF CONCLUSION TARGET VECTOR:
            
            ##METHOD 1:
            ###build combination of combination_of from top_k, map their average to the new space, then find the ponit that is
            ###closest to their average...
            if combination_of <= len(top_p_target_vectors):
                premise_targets_combinations = list(itertools.combinations(top_p_target_vectors, combination_of))
            else:
                premise_targets_combinations = [top_p_target_vectors]

            mapped_avg_vectors = [model.get_avg_embedding(torch.from_numpy(np.mean(x, axis=0))).detach().numpy() for x in premise_targets_combinations]
            avg_mapped_vec = np.mean(mapped_avg_vectors, axis=0)

            ##METHOD 2:
            ###SIMPLY MAP THE AVERAGE OF TOP PREMISE TARGETS...
            #avg_mapped_vec = model.get_avg_embedding(torch.from_numpy(avg_vec))
        
            ##METHOD 3:
            ###MAP EACH PREMISE TARGET OF THE TOP K INDIVIDUALLY TO THE NEW SPACE AND THEN AVERAGE THEM
            #mapped_top_premise_vectors = [model.get_target_embedding(torch.from_numpy(x)).detach().numpy() 
            #                          for x in top_p_target_vectors]
            #avg_mapped_vec = np.mean(mapped_top_premise_vectors[0:combination_of], axis=0)
            
            # 4. Find the closest target from the lexicon to the mapped average of premise targets..
            model_scores = [distance.cosine(avg_mapped_vec, x) for x in all_mapped_vectors]
            model_scores = [2 if np.isnan(x) else x for x in model_scores]
            model_pred = all_targets[np.argmin(model_scores)]
            
            if input_texts is not None:
                #print('Enabling advanced model...')
                if compute_overlap(input_texts[idx], model_pred) == 0:
                    print(model_pred)
                    model_pred = top_p_targets[0][0]
                    print('replaced with:', model_pred)
            idx+=1

            predicted_targets.append(model_pred)
            baseline_predicted_targets.append(simple_avg_pred)
            true_targets.append(true_conc_target)
            
            for threshold in thresholds:
                if compute_overlap(true_conc_target, model_pred) >= threshold:
                    model_correct_cases_at_threshold[threshold]+=1
                if compute_overlap(true_conc_target, simple_avg_pred) >= threshold:
                    baseline_correct_cases_at_threshold[threshold]+=1


        model_bleu_score    = eval_bleu(predicted_targets, true_targets)
        model_meteor_score  = eval_meteor(predicted_targets, true_targets)
        baseline_bleu_score = eval_bleu(baseline_predicted_targets, true_targets)
        baseline_meteor_score = eval_meteor(baseline_predicted_targets, true_targets)

        baseline_acc_at_thresholds = [round(x[1]/len(test_data), 2) for x in baseline_correct_cases_at_threshold.items()]
        model_acc_at_thresholds = [round(x[1]/len(test_data), 2) for x in model_correct_cases_at_threshold.items()]
        
        return  baseline_acc_at_thresholds, model_acc_at_thresholds, \
                baseline_meteor_score, baseline_bleu_score,\
                model_meteor_score, model_bleu_score,\
                predicted_targets, baseline_predicted_targets