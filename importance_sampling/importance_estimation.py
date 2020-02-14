import json
import random
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import normalize, MinMaxScaler

import nltk
from   nltk.corpus import sentiwordnet as swn
from   matplotlib import pyplot as plt

from lexrank import STOPWORDS, LexRank


class ImportanceEstimationModel(object):

    def load_data(self, train_path, dev_path, test_path):
        train_data = json.load(open(train_path))
        dev_data   = json.load(open(dev_path))
        test_data  = json.load(open(test_path))

        self.train_data = train_data
        self.test_data  = test_data
        self.dev_data   = dev_data

        return self.train_data, self.dev_data, self.test_data

    def print_train_sample(self):
        sample = random.choice(self.train_data)
        claims = sample[0]
        conc   = sample[1]
        print('Conclusion:', conc)
        print('Claims:')
        for idx, claim in enumerate(claims):
            print(idx+1, '.', claim)

    def num_of_pos_tags_feature(self, claim):
        claim_annotated = self.nlp_parser.sentences_to_tags([claim])
        claim_pos_tags = set([x[1] for x in claim_annotated[0]])
        return len(claim_pos_tags)

    def num_of_ne_feature(self, claim):
        named_entities = self.nlp_parser.extract_named_entities(claim)
        return len(named_entities)


    def _build_tfidf_model(self, texts):
        tfidf = TfidfVectorizer()
        tfidf_model = tfidf.fit(texts)
        return tfidf_model

    def _build_lexrank_model(self, texts):
        self.lxrank = LexRank(texts, stopwords=STOPWORDS['en'])

    def _sentiment_features(self, claim):
        claim_words = nltk.word_tokenize(claim)

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
        
        return num_of_positive_words, num_of_negative_words , num_of_neutral_words


    def _num_of_words_feature(self, claim):
        claim_words = nltk.word_tokenize(claim)
        return len(claim_words)

    def _tfidf_features(self, claim):
        claim_words = nltk.word_tokenize(claim)
        
        # Avg. tfidf
        tfidf_vector = self.tfidf_model.transform([claim])
        avg_tfidf_feature = np.sum(tfidf_vector.toarray())/len(claim_words)
        max_tfidf_feature = np.max(tfidf_vector.toarray())

        return avg_tfidf_feature, max_tfidf_feature

    def _claim_features(self, claim, claims_text):

        # Number of words
        num_of_words_feature = self._num_of_words_feature(claim['text'])
        
        # Avg. Max. tfidf
        avg_tfidf_feature, max_tfidf_feature = self._tfidf_features(claim['text'])
        
        # Number of postive/negative/neutral words
        num_of_positive_words, num_of_negative_words , num_of_neutral_words  = self._sentiment_features(claim['text'])
        
        # Number of POS tags and Number of Named Entities
        poss = set([p['type'] for p in claim['pos']])
        num_of_pos_tags = len(poss)
        num_of_ne  = len(claim['named_entities'])


        return [num_of_words_feature, 
                avg_tfidf_feature, 
                max_tfidf_feature,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words,
                num_of_ne, num_of_pos_tags]

    def _instance_features(self, claims):
        claims_sents = [claim['text'] for claim in claims] 
        claims_text  = ' '.join(claims_sents)

        claims_centroidness_scores = self.lxrank.rank_sentences(claims_sents, threshold=None, fast_power_method=False)
        claims_features = [self._claim_features(claim, claims_text) + claims_centroidness_scores[i] for i, claim in enumerate(claims)]

        return np.atleast_2d(claims_features)

    def instance_scores(self, claims, summary):
        claims_labels = []
        for claim in claims:
            claim_tokens   = set(nltk.word_tokenize(claim['text']))
            summary_tokens = set(nltk.word_tokenize(summary))

            shared_tokens = claim_tokens.intersection(summary_tokens)

            #overlap_ratio = len(shared_tokens)/(len(claim_tokens) + len(summary_tokens))
            
            claims_labels.append(len(shared_tokens))
        
        return claims_labels

    def feature_representation(self, data):
        # 1. build a tf-idf model over the training data
        arguments = [' '.join([claim['text'] for claim in argument['claims']]) for argument in data]
        self.tfidf_model = self._build_tfidf_model(arguments)

        arguments = [[claim['text'] for claim in argument['claims']] for argument in data]
        self.lxrank_model = self._build_lexrank_model(arguments)

        # 2. Encode training data into features
        self.train_X = []
        self.train_Y = []

        for argument in data:
            claims     = argument['claims']
            conclusion = argument['conclusion']['text']

            claims_vectors = self._instance_features(claims)
            claims_scores  = self.instance_scores(claims, conclusion)

            for claim_vector, claim_label in zip(claims_vectors, claims_scores):
                self.train_X.append(claim_vector)
                self.train_Y.append(claim_label)


        self.train_X = np.array(self.train_X)
        self.train_Y = np.array(self.train_Y)

        #Normalize claims_scores into [0,1]
        labels_scaler = MinMaxScaler()
        labels_scaler.fit(self.train_Y.reshape(-1, 1))
        self.train_Y = labels_scaler.transform(self.train_Y.reshape(-1, 1)).reshape(-1)

        return self.train_X, self.train_Y

    def train_svr(self, train_X, train_Y):
        svr_params = {'C': [0.001, 0.1, 1.0, 10, 100]}
        svr = SVR()
        
        clf = GridSearchCV(svr, svr_params, cv=5, scoring='neg_mean_absolute_error', return_train_score=False)
        clf.fit(train_X, train_Y)

        best_ridge = clf.best_estimator_
        
        self.best_ridge = best_ridge

        return clf.best_score_


    def kendalltau_evaluation(self, test_data):
        from scipy import stats
        
        total_tau = 0
        for sample in test_data:
            claims     = sample[0]
            conclusion = sample[1]

            #Predict scores of each claim
            claims_vectors = self._instance_features(claims)
            ground_truth_scores  = self.instance_scores(claims, conclusion)
            ground_pred_scores  = self.best_ridge.predict(claims_vectors)

            tau, _ = stats.kendalltau(ground_truth_scores, ground_pred_scores)
            total_tau += tau

        return total_tau/len(test_data)

    def score_data(self, data):
        for sample in data:
            claims     = sample['claims']
            conclusion = sample['conclusion']['text']

            #Predict scores of each claim
            claims_vectors = self._instance_features(claims)
            claims_scores  = self.best_ridge.predict(claims_vectors)

            for claim, score in zip(claims, claims_scores):
                claim['imprtance_score'] = score

            sample['claims'] = claims

        return data

    def mrr_evaluation(self, test_data):
        mrr_value = 0
        for sample in test_data:
            claims     = sample['claims']
            conclusion = sample['conclusion']['text']

            #Predict scores of each claim
            claims_vectors = self._instance_features(claims)
            claims_labels  = self.instance_scores(claims, conclusion)
            claims_labels  = [c_score > 0 for c_score in claims_labels]

            claims_scores  = self.best_ridge.predict(claims_vectors)

            #Sort claims based on the score
            scores_labels_list = list(zip(claims_scores, claims_labels))
            sorted_claims = sorted(scores_labels_list, key= lambda x : -x[0])

            rank = 1
            for x in  sorted_claims:
                if x[1]:
                    break
                rank +=1

            mrr_value += 1/rank

        return mrr_value/len(test_data)
