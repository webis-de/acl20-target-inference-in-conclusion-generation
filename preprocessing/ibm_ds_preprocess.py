import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)

import json
import nltk
import csv
import glob
import math
import spacy
import numpy as np

from utils.nlp_parser import NlpParser


output_path = '../target_identification/data/ibm_ds'

#np_grammar = "NP: {<DT>?<JJ>*<NN>}"
#parser = nltk.RegexpParser(np_grammar)


def creat_learning_files(topics, output_path):

    no_topics = len(topics)
    no_training_topics = math.ceil(no_topics * 0.7)
    no_dev_topics = math.ceil(no_topics * 0.2)
    no_test_topics = math.ceil(no_topics * 0.1)

    print('num of topics:', no_topics)
    print('num of training topics', no_training_topics)
    print('num of dev topics', no_dev_topics)
    print('num of test topics', no_test_topics)
    
    training_claims = []
    for i in range(0, no_training_topics):
        topic = topics[i]
        for claim in topic:
            training_claims += claim
            training_claims.append(('', '', ''))
            
    with open(output_path + '/train.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(training_claims)


    dev_claims = []
    for i in range(no_training_topics, no_training_topics + no_dev_topics ):
        topic = topics[i]
        for claim in topic:
            dev_claims += claim
            dev_claims.append(('', '', ''))
            
    with open(output_path + '/dev.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(dev_claims)

    test_claims = []
    for i in range(no_training_topics + no_dev_topics, len(topics) ):
        topic = topics[i]
        for claim in topic:
            test_claims += claim
            test_claims.append(('', '', ''))
            
    with open(output_path + '/test.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(test_claims)

def annotate_ibm_with_cornlp(path):
    json_obj = json.load(open(path))

    nlpParser = NlpParser()

    train_topics = []
    test_topics  = []
    for t_idx, topic in enumerate(json_obj):
        topic_id = topic['topicId']
        topic_split = topic['split']
        claims   = topic['claims']

        
        print('process topic:', topic_id)

        #filter non compitable claims
        claims = list(filter(lambda x: x['Compatible'] != 'no', claims))
        claims_txts = list(map(lambda x: x['claimCorrectedText'], claims))
        claims_targets = list(map(lambda x: nltk.word_tokenize(x['claimTarget']['text']), claims))
        tagged_claims = nlpParser.parse_sents(claims_txts)

        topic_claims = []
        for claim_idx, tagged_claim in enumerate(tagged_claims):
            claim_target = claims_targets[claim_idx]
            claim_ann = list(map(lambda x: [x[0], x[1], x[2], 'O'], tagged_claim))
            
            #Adding the ct tags
            for i in range(0, len(claim_ann)):
                temp = tuple(claim_ann[i:i+len(claim_target)])
                temp_tokens = tuple(list(map(lambda x: x[0], temp)))
                if tuple(temp_tokens) == tuple(claim_target):
                    claim_ann[i][3]= 'B-CT'
                    for j in range(i+1, i+len(claim_target)):
                        claim_ann[j][3]= 'I-CT'

            topic_claims.append(claim_ann)
        
        if topic_split == 'test':
            test_topics.append(topic_claims)
        else:
            train_topics.append(topic_claims)

    #shuffle the data
    train_claims = [claim for topic_claims in train_topics for claim in topic_claims]
    np.random.shuffle(train_claims)
    test_claims  = [claim for topic_claims in test_topics for claim in topic_claims]
    np.random.shuffle(test_claims)

    #save data
    with open('../ibm_debater_cs/annotated/train.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for c in train_claims:            
            for t in c:
                writer.writerow((t[0].strip(), t[1], t[3]))
            writer.writerow(('','',''))

    with open('../ibm_debater_cs/annotated/test.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for c in test_claims:
            for t in c:
                writer.writerow((t[0].strip(), t[1], t[3]))
            writer.writerow(('','',''))

#annotate_ibm_with_cornlp('../ibm_debater_cs/claim_stance_dataset_v1.json')
#creat_learning_files('../ibm_debater_cs/annotated', '../ibm_debater_cs/annotated')

