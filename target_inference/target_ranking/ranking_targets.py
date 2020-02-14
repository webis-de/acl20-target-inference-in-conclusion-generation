import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)


from nltk.corpus import sentiwordnet as swn
from sklearn.metrics import pairwise
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.svm import SVR
import gensim.downloader as api
import numpy as np
from sklearn.externals import joblib
import argparse
import json
import pickle
import nltk

from   nltk.corpus import sentiwordnet as swn

import conclusion_model.utils as utils


def _sentiment_features(claim):
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


def best_entity(entities):
    sorted(entities, key=lambda x: -x['confidence'])
    return entities[0]

def compute_overlap(target_txt, conc_target_txt, binary=True):
    from nltk.corpus import stopwords
    swords = stopwords.words('english')
    target_tokens = set([x for x in target_txt.strip().lower().split(' ') if x not in swords ])
    conc_tokens   = set([x for x in conc_target_txt.strip().lower().split(' ') if x not in swords])    
    #print(target_tokens)
    #print(conc_tokens)

    shared_tokens = target_tokens.intersection(conc_tokens)
    all_tokens    = target_tokens.union(conc_tokens)
    #print(overlap_ratio)
    
    return len(shared_tokens)/len(all_tokens) if len(all_tokens) > 0 else 0 #1 if overlap_ratio >= 0.2 else 0


def target_w2v(emb_model, target):
    target_tokens = target.split(' ')
    target_tokens = list(filter(lambda x: x in emb_model, target_tokens))
    targets_vecs  = [emb_model[token] for token in target_tokens]
    if len(targets_vecs) == 0:
        return np.zeros(300)
    else:
        target_vec = np.mean([emb_model[token] for token in target_tokens], axis=0)
        return target_vec

def target_features(targets_sim_matrix, target_idx, target, claim):
    # Avg similarity to other targets
    f_sim = np.mean(targets_sim_matrix[target_idx])

    # Number of words
    f_num_words = len(target['text'].split(' '))
    # Start pos and End pos
    f_spos = target['start_pos']
    f_epos = target['end_pos']

    # Confidence of the tagger annotator
    f_tag_conf = target['confidence']

    # Claim length
    f_clen = len(claim.split(' '))

    # Claim sentiment
    num_of_positive_words, num_of_negative_words , num_of_neutral_words  = _sentiment_features(claim)
    
    return [f_sim, f_spos, f_epos, f_num_words, f_tag_conf, f_clen, num_of_positive_words, num_of_negative_words , num_of_neutral_words]

def build_instance(claims, conclusion_target_txt=None):
    # Filter out claims that doesnt contain entities
    claims = list(filter(lambda x: len(x['targets']) > 0, claims))
    
    if len(claims) == 0:
            return []

    targets = [(target, claim['text']) for claim in claims for target in claim['targets']]
    # Compute pair-wise similarity
    targets_vectors    = [utils.glove_embed(target[0]['text']) for target in targets]
    targets_sim_matrix =  pairwise.cosine_similarity(targets_vectors, targets_vectors)
    
    instances = []    
    for idx, item in enumerate(targets):
        target     = item[0]
        claim_text = item[1]
        
        features = target_features(targets_sim_matrix, idx, target, claim_text)

        if conclusion_target_txt != None :
            label = compute_overlap(target['text'], conclusion_target_txt)
            instances.append(features + [label])
        else:
            instances.append(features)
        
    return instances
    
def conclusion_target(conc):
    return conc['text'] if len(conc['targets']) == 0 else conc['targets'][0]['text']


def train_lambdamart(data_path, output_path):
    import pyltr

    data = json.load(open(data_path, 'r'))

    arguments = list(map(lambda x: build_instance(x['claims'], conclusion_target(x['conclusion'])), data))
    instances = [(instance, idx) for idx, argument in enumerate(arguments) for instance in argument]
    
    features  = np.array([instance[0][0:-1] for instance in instances])
    labels    = np.array([instance[0][-1] for instance in instances])
    qIdxs     = np.array([instance[1] for instance in instances])


    print(features.shape)
    print(labels.shape)
    print(qIdxs.shape)
    # Normalization...
    # feats_scaler  = MinMaxScaler()
    # feats_scaler.fit(features)
    # labels_scaler = MinMaxScaler()
    # labels_scaler.fit(np.array(labels).reshape(-1, 1))

    # norm_features = feats_scaler.transform(features)
    # norm_labels   = labels_scaler.transform(np.array(labels).reshape(-1, 1))
    # norm_labels   = norm_labels.reshape(-1)

    train_x, test_x, train_y, test_y, train_qidx, test_qidx = train_test_split(features, labels, qIdxs, test_size=0.2, shuffle=False)

    #Training
    metric  = pyltr.metrics.NDCG(k=10)
    monitor = pyltr.models.monitors.ValidationMonitor(test_x, test_y, test_qidx, 
                                                        metric=metric, stop_after=500)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=1,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )

    print('start training...')
    model.fit(train_x, train_y, train_qidx, monitor=monitor)

    test_pred = model.predict(test_x)
    print('Random ranking:', metric.calc_mean_random(test_qidx, test_y))
    print('Our model:', metric.calc_mean(test_qidx, test_y, test_pred))

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

def train_model(emb_model, data_path, output_path):
    data = json.load(open(data_path, 'r'))

    instances = list(map(lambda x: build_instance(emb_model, x['claims'], x['conclusion']['text']), data))
    instances = [instance for sublist in instances for instance in sublist]
    features  = [instance[0:-1] for instance in instances]
    labels    = [instance[-1] for instance in instances]


    # Normalization...
    feats_scaler  = MinMaxScaler()
    feats_scaler.fit(features)
    labels_scaler = MinMaxScaler()
    labels_scaler.fit(np.array(labels).reshape(-1, 1))

    norm_features = feats_scaler.transform(features)
    norm_labels   = labels_scaler.transform(np.array(labels).reshape(-1, 1))
    norm_labels   = norm_labels.reshape(-1)

    train_x, test_x, train_y, test_y = train_test_split(norm_features, norm_labels, test_size=0.2)

    svr_parameters = {'kernel':('poly', 'rbf'), 'C':[1, 10]}
    svr = SVR()
    clf = GridSearchCV(svr, svr_parameters, cv=5, scoring='neg_mean_absolute_error', return_train_score=False)
    clf.fit(train_x, train_y)

    print('best score:', clf.best_score_)

    best_svc = clf.best_estimator_

    print('saving model...')
    joblib.dump(best_svc, output_path + 'ranking.model')
    joblib.dump(feats_scaler, output_path + 'feats_scaler.model')

def generate_training_file(path, output_path):
    data = json.load(open(path))

    with open(output_path, 'w') as f:
        for sample in data:
            premise_targets   = [target['text'].lower() for claim in  sample['claims'] for target in claim['targets']]
            conclusion_target = sample['conclusion']['targets'][0]['text'].lower() if len(sample['conclusion']['targets']) > 0 else sample['conclusion']['text'].lower()
            overlaps = [str(compute_overlap(t, conclusion_target)) for t in premise_targets]

            if len(premise_targets) == 0 or len(conclusion_target) == 0:
                continue

            f.write(','.join(premise_targets) + '\t' + conclusion_target + '\t' + ','.join(overlaps) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('task', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)



    args = parser.parse_args()

    #generate_training_file(args.data_path, args.output_path)

    if args.task == 'train':
        train_lambdamart(args.data_path, args.output_path)
    elif args.task == 'simple_scoring':
        score_targets_based_on_similarity(args.data_path, args.output_path)
    else:
        apply_scoring(args.data_path, args.model_path, args.output_path)


# Train: python ranking_targets.py train ./ ./ ../data/idebate/train.json
# Scoring: python ranking_targets.py score ./ ../data/idebate/test_scored.json ../data/idebate/test.json
# Simple scoring: python ranking_targets.py simple_scoring ./ ../data/idebate/test_scored.json ../data/idebate/test_scored.json