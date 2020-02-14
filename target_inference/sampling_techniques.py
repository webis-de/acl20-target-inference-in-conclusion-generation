# -*- encoding: utf-8 -*-
import sys
import os
import pickle
import itertools
import nltk
import numpy as np
from scipy.spatial import distance
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')


nb_dir = os.path.split(os.getcwd())[0]
np_dir = nb_dir.replace('target_inference', '')
sys.path.append(nb_dir)


from target_inference import utils as utils

def compute_overlap(predicted_target, true_target):
    from nltk.corpus import stopwords

    true_target = true_target.strip().lower().split(' ')
    predicted_target = predicted_target.strip().lower().split(' ')

    english_stopwords = stopwords.words('english')

    true_target      = set([t for t in true_target if t not in english_stopwords])
    predicted_target = set([t for t in predicted_target if t not in english_stopwords])

    shared_tokens = true_target.intersection(predicted_target)
    all_tokens    = true_target.union(predicted_target)

    return len(shared_tokens)/len(all_tokens) if len(all_tokens) > 0 else 0


def triple_sampling(conc_target, premise_targets, target_space={}, embedding_method_name='glove', combination_of=2, num_neg_cases=1):
    """
    triple sampling...
    """
    #print('triple sampling technique..')
    samples = []
    negative_targets = {}
    overlaps = [compute_overlap(x[0], conc_target[0]) for x in premise_targets]
    max_overlap = np.max(overlaps)
    min_overlap = np.min(overlaps)

    conc_vector = utils.embed_sentence(conc_target[0], conc_target[1], embedding_method_name=embedding_method_name)
    premise_vectors = [utils.embed_sentence(x[0], x[1], embedding_method_name=embedding_method_name) for x in premise_targets]


    positive_instances = []
    negative_instances = []
    anchor_instances   = [conc_vector]
    #print('Anchor conc: ', conc_target[0])
    #print('Pos is average of premise targets..')
    
    

    # if max_overlap > 0.4:
    #     cand_vec = premise_vectors[np.argmax(overlaps)]
    #     anchor_instances.append(cand_vec)


    #combination of combination_of premise targets
    if combination_of <= len(premise_vectors):
        premise_targets_combinations = list(itertools.combinations(premise_vectors, combination_of))
    else:
        premise_targets_combinations = [premise_vectors]

    average_vectors = [np.mean(x, axis=0) for x in premise_targets_combinations]
    
    # if np.all(avg_vector == 0): #If the average vector of the premise targets is zeros..
    #     return []

    #find positive case
    for average_vector in average_vectors:
        if not np.array_equal(average_vector, conc_vector):
            positive_instances.append(average_vector)

    if target_space != {}:
        cand_targets = list(target_space.keys())
        for i in range(0, num_neg_cases):
            cand_target = np.random.choice(cand_targets)
            cand_vector = target_space[cand_target]
            negative_instances.append(cand_vector)
            negative_targets[cand_target] = cand_vector
            #print('Neg random: ', cand_target)


    for anchor_instance in anchor_instances:
        for pos_inst in positive_instances:
            anchor_to_pos_distance = distance.cosine(anchor_instance, pos_inst)
            for neg_inst in negative_instances:
                anchor_to_neg_distance = distance.cosine(anchor_instance, neg_inst)
                #if anchor_to_neg_distance > anchor_to_pos_distance :               
                samples.append((anchor_instance, pos_inst, neg_inst))

    return samples, [(x, avg_vector) for x in premise_vectors for avg_vector in average_vectors], negative_targets
