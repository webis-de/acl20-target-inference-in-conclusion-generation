import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)


import numpy as np
import gensim.downloader as api
import nltk
import json
import random
import torch

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from scipy.spatial import distance

from utils import *
from siamese_nn import models

import ranking_targets
import pickle

def claims_features_to_targets(text):
    entities = {}
    idx = -1
    for token in text.split():
        word, tag = token.split('￨')
        if tag == 'B-CT':
            idx+=1
            entities[idx] = [word]
        if tag == 'I-CT':
            entities[idx].append(word)

    return [' '.join(x) for x in entities.values()]

def best_entity(entities):
    if entities == []:
        return None
    sorted(entities, key=lambda x: -x['confidence'])
    return entities[0]['text']

def premise_sentiment(premise):
    claim_words = nltk.word_tokenize(premise)

    num_of_positive_words = 0
    num_of_negative_words = 0
    num_of_neutral_words  = 0
    for word in claim_words:
        synsets = list(swn.senti_synsets(word))
        if len(synsets) == 0:
            num_of_neutral_words +=1
        else:
            syn = synsets[0]
            if syn.pos_score() > syn.neg_score():
                num_of_positive_words +=1
            elif syn.pos_score() < syn.neg_score():
                num_of_negative_words +=1
            else:
                num_of_neutral_words+=1
    

    return num_of_positive_words - num_of_negative_words

class ClaimTargetModeling(object):
    def get_target(self, premises):
        return 'Not implemented'

    def get_sentiment(self, premises):
        return 'Not implemented'

    def gen_conclusion(self, premises):
        return 'Not implemented'

class RankedTargetModel(ClaimTargetModeling):
    
    def __init__(self, ranking_model_path, load_concepts=True):
        self.ranking_model = pickle.load(open(ranking_model_path, 'rb'))

    def get_random_target(self, premises):
        random.shuffle(premises)
        targets = premises[0]['targets']
        idx = 1
        while idx < len(premises) and len(targets) == 0:
            targets = premises[idx]['targets']
            idx +=1
        
        #sort targets based on confidence
        targets = sorted(targets, key=lambda x: -x['confidence'])

        #print(targets)
        if len(targets) == 0:
            print('failed to generate target..')
            return 'UNKOWN'
        else:
            return targets[0]['text']

    def get_top_k_targets(self, premises, k=1):
        targets = [(target, p['text']) for p in premises for target in p['targets']]

        if targets == []:
            return []

        targets_features = ranking_targets.build_instance(premises)
        targets_scores = self.ranking_model.predict(targets_features)

        sorted_targets_by_scores = sorted(list(zip(targets, targets_scores)), key=lambda x: -x[1])

        return [(x[0][0]['text'], x[0][1]) for x in sorted_targets_by_scores[0:k]]

    def get_target(self, premises):
        targets = [(target, p['text']) for p in premises for target in p['targets']]

        if targets == []:
            return 'UNKOWN'

        targets_features = ranking_targets.build_instance(premises)
        targets_scores = self.ranking_model.predict(targets_features)

        #print(targets_scores)
        #print(targets)

        return targets[np.argmax(targets_scores)][0]['text']

    def _conc_text(self, target, sent):
        if sent == 0:
            temp = np.random.choice(neu_templates)
        elif sent > 0:
            temp = np.random.choice(pos_templates)
        else:
            temp = np.random.choice(neg_templates)

        return temp.replace('target', target)
    
    def get_sentiment(self, premises):
        if type(premises) == list:
            sents = [premise_sentiment(x['text']) for x in premises]
            avg_sents = np.mean(sents)
            return avg_sents
        else:
            return premise_sentiment(x)

    def gen_conclusion(self, premises):
        conc_target = self.get_target(premises)
        conc_sent   = self.get_sentiment(premises)

        return self._conc_text(conc_target, conc_sent)


class TargetSpaceModeling(ClaimTargetModeling):
    
    def __init__(self, siamese_model_path=None, load_concepts=True, embedding_method='glove', src_target_space_path=None):
        self.stopwords = stopwords.words('english')
        self.embedding_method = embedding_method

        if load_concepts:
            self.builtin_concepts = set(open('../data/concepts.txt').read().split('\n')[:-1])
        else:
            self.builtin_concepts = set()

        if src_target_space_path != None:
            print('Loading src target space from:', src_target_space_path)
            self.all_targets  = json.load(open(src_target_space_path + '.json'))
            self.src_target_space = pickle.load(open(src_target_space_path + '_space.pkl', 'rb'))
        else:
            self.all_targets = []
            self.src_target_space = {}

        if siamese_model_path != None:
            print('Mapping target space using:', siamese_model_path)
            self.model = models.get_triplelet_model(siamese_model_path)
            self.map_target_to_new_space()
            print('Size of Target Space:', len(self.learned_target_space))
        else:
            self.model = None

    def _accepted_target(self, target):
        return (target != None) and (target not in self.stopwords) and (len(target.split(' ')) < 5)

    def map_target_to_new_space(self):        
        self.learned_target_space = {}
        for chunk in chunks(list(self.src_target_space.items()), 10): #transform every 10 vectors at once
            targets, vectors = zip(*chunk)
            tensor = torch.from_numpy(np.array(vectors))
            mapped_tensors = self.model.get_target_embedding(tensor).detach().numpy()

            for target, mapped_tensor in zip(targets, mapped_tensors):
                self.learned_target_space[target] = mapped_tensor #/np.linalg.norm(mapped_tensor)

        print('Finshed Mapping...')


    def fit_on_data(self, train_data, normalize=True):
        premises = [] #[premise for x in train_data for premise in x['claims']]
        conclusions = [x['conclusion'] for x in train_data]

        train_targets = set([(target, s['text']) for s in (premises + conclusions) for target in self.get_premise_targets(s)])
        train_targets = set([target for target in train_targets if self._accepted_target(target[0])])

        self.builtin_concepts = set([(x,x) for x in self.builtin_concepts]) # To match with the structure of claims DS...

        self.all_targets = train_targets | self.builtin_concepts

        print('Number of targets:', len(self.all_targets))
        self.src_target_space = {target[0]: embed_sentence(target[0], target[1], normalize=normalize,  embedding_method_name = self.embedding_method) for target in self.all_targets if target != None}
        #filter none targets
        self.src_target_space = dict([(key, value) for key, value in self.src_target_space.items() if value is not None])
        self.all_targets = [target for target in self.all_targets if target[0] in self.src_target_space]

        if self.model != None:
            self.learned_target_space = {
                        target[0]: self.model.get_target_embedding(torch.from_numpy(self.src_target_space[target[0]]).unsqueeze(1).unsqueeze(0)).detach().numpy().reshape(-1) 
                        for target in self.all_targets if target != None}
        else:
            self.learned_target_space = {}


    def _conc_text(self, target, sent):
        if sent == 0:
            temp = np.random.choice(neu_templates)
        elif sent > 0:
            temp = np.random.choice(pos_templates)
        else:
            temp = np.random.choice(neg_templates)

        return temp.replace('target', target)

    def get_premise_targets(self, premise):
        return [target['text'] for target in premise['targets']]

    def get_premise_target(self, premise):
       
        if len(premise['targets']) == 0:
            return None


        sorted_targets = sorted(premise['targets'], key=lambda x: -x['confidence'])
        
        premise_target = sorted_targets[0]['text']

        return premise_target.lower()

    def score_target(self, target_text, target_vec, premises, a=0.5, b=0.5):
        avg_sim_to_premises = np.mean([(1-distance.cosine(target_vec, premise[1])) for premise in premises if premise[0] != target_text])
        target_num_words = len(target_text.split(' '))

        score = a * avg_sim_to_premises + b * (1/target_num_words)

        return score

    def get_target(self, premises, simple_avg=False):
        targets = [(target['text'], p['text']) for p in premises for target in p['targets']]
        targets = list(map(lambda x: (x[0], embed_sentence(x[0], x[1], embedding_method_name=self.embedding_method)), targets))

        #print(targets)
        if len(targets) == 0:
            return 'UNKOWN'

        premise_targets, premise_target_vectors = zip(*targets)
        conc_vector = np.mean(premise_target_vectors, axis=0)
        
        if simple_avg:
            #create lexicon of all concepts we know plus the new premises' targets
            target_lexicon = list(self.src_target_space.items()) + targets
            #target_lexicon = targets
            scores = [(distance.cosine(conc_vector, vector)) for _, vector in target_lexicon]
            scores = [2 if np.isnan(x) else x for x in scores]
        else:

            #Map the target into the learned space
            tensor = torch.from_numpy(np.array(premise_target_vectors))
            mapped_target_vectors = self.model.get_target_embedding(tensor).detach().numpy()
            mapped_targets = [(target, vector) for target, vector in zip(premise_targets, mapped_target_vectors)]
            # mapped_conc_vector is average of the mapped premise targets...
            mapped_conc_tensor = np.mean(mapped_target_vectors, axis=0)

            # OR it is mapping of the average vector of the premise targets...
            #Map the conclusion (avg) vector to the learned space
            #conc_tensor = torch.from_numpy(conc_vector).unsqueeze(0)
            #mapped_conc_tensor = self.model.get_target_embedding(conc_tensor)
            #mapped_conc_tensor = mapped_conc_tensor.detach().numpy().reshape(-1)
            #Normalizing..
            #mapped_conc_tensor = mapped_conc_tensor/np.linalg.norm(mapped_conc_tensor) if np.linalg.norm(mapped_conc_tensor) != 0 else mapped_conc_tensor
            

            #create lexicon of all concepts we know plus the new premises' targets            
            target_lexicon = list(self.learned_target_space.items()) + mapped_targets
            #target_lexicon = mapped_targets
            scores = [distance.cosine(x[1], mapped_conc_tensor) for x in target_lexicon]
            scores = [2 if np.isnan(x) else x for x in scores]

        # print(scores)
        # print(target_lexicon)
        return target_lexicon[np.argmin(scores)][0]

        
    def gen_conclusion(self, instance):
        conc_target = self.get_target(instance['claims'])
        conc_sent   = premise_sentiment(instance['claims'])

        return self._conc_text(conc_target, conc_sent)