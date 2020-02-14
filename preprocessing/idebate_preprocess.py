import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)

import json
import numpy as np
import random
import itertools
import argparse
from functools import partial

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from utils.nlp_parser import *

def findsubsets(S,m):
    return itertools.combinations(S, m)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def annotate_item(item):
    c_text = item['text']
    item_entities = sorted(item['targets'], key=lambda x: -x['confidence'])
    if len(item_entities) > 0:
        substring = item_entities[0]['text']
        new_substring = ' '.join(list(map(lambda x: x[1] + '￨B-CT' if x[0] == 0 else x[1] + '￨I-CT', 
                                      enumerate(substring.split(' ')))))
        
        c_text = c_text[:item_entities[0]['start_pos']] + new_substring + c_text[item_entities[0]['end_pos']:]  #c_text.replace(substring, new_substring)

    c_text = list(map(lambda x: x if ('￨B-CT' in x or '￨I-CT' in x) else x + '￨O', c_text.split(' ')))
    c_text = ' '.join(c_text)
    
    return c_text

def process_instance(claims):
    src_claims = []
    for claim in claims:
        src_claims.append(annotate_item(claim))

    return src_claims

def create_mask(claims):
    tokens = []
    for claim in claims:
        ann_claim = annotate_item(claim)
        ann_claim = ann_claim.split(' ')

    return {'logits': logits, 'words': words}

def top_k_claims(claims, scoring_field='target_score', k=5):
    sorted_claims = sorted(claims, key=lambda x: -x[scoring_field])
    return sorted_claims[0:k]

def extract_conc(item):
    conclusion_text = '<SCONC> ' +  item['conclusion']['text'] + ' <ECONC>'
    if len(item['conclusion']['targets']) >0:
        conc_entities   = sorted(item['conclusion']['targets'], key=lambda x: -x['confidence'])
        conc_entity     = '<SCONC> ' + conc_entities[0]['text'] + ' <ECONC>'
    else:
        conc_entity = '<NONE>'

    return conclusion_text, conc_entity

#Always get the top k claims...
def test_to_summary_file(data, k=5):
    resulted_data = []
    resulted_mask = []

    for item in data:

        #get top k claims
        claims = top_k_claims(item['claims'], scoring_field='imprtance_score', k=k)
        
        conclusion_text, conc_entity = extract_conc(item)
        
        claims_with_features = ' <EOC>￨S '.join(process_instance(claims))
        claims_texts         = ' <EOC> '.join(list(map(lambda x: x['text'], claims)))

        # Mask over the claims
        claims_tokens = claims_with_features.split(' ')
        claims_mask   = [1 if '-CT' in token else 0 for token in claims_tokens]
        resulted_mask.append({'logits': claims_mask, 'words': claims_tokens})
        resulted_data.append((claims_texts, claims_with_features, conclusion_text, conc_entity))
        
    return resulted_data, resulted_mask


def collect_samples(items, num_of_samples, sample_size, importance_field=None):
    result = []

    if importance_field != None:
        items_probabilities = np.array([x[importance_field] for x in items])
        #Normalize between 0 and 1
        items_probabilities = softmax(items_probabilities)
    else:
        items_probabilities = softmax(np.array([1 for x in items]))

    for i in range(0, num_of_samples):
        sample = np.random.choice(items, sample_size, replace=False, p=items_probabilities)
        result.append(sample)

    return result

def train_to_summary_file(data, k=5, only_top_claims=False, max_choice=5):
    resulted_data = []
    resulted_mask = []

    for item in data:
        # if len(item['conclusion']['targets']) == 0:
        #     print('skipping empty conclusion target..') 
        #     continue

        #get the claims
        claims = item['claims']
        #filter only claims with targets
        #claims = list(filter(lambda x: len(x['targets']) > 0, claims))
        
        conclusion_text, conc_entity = extract_conc(item)

        if claims == []:
            print('empty claims... skip!!!')
            continue
        if len(claims) < k:
            claims = top_k_claims(claims, scoring_field='imprtance_score', k=k)

            claims_with_features = ' <EOC>￨S '.join(process_instance(claims))
            claims_texts         = ' <EOC> '.join(list(map(lambda x: x['text'], claims)))

            # Mask over the claims
            claims_tokens = claims_with_features.split(' ')
            claims_mask   = [1 if '-CT' in token else 0 for token in claims_tokens]
            resulted_mask.append({'logits': claims_mask, 'words': claims_tokens})

            resulted_data.append([claims_texts, claims_with_features, conclusion_text, conc_entity])
        else:
            if only_top_claims:
                claims_subsets = [top_k_claims(claims, scoring_field='imprtance_score', k=k)]
            else:
                #Sample max_choice subsets of size k 
                claims_subsets = collect_samples(claims, max_choice, k, 'imprtance_score') #list(findsubsets(claims, k))
            
            if max_choice < len(claims_subsets):
                choices = np.random.choice(len(claims_subsets), max_choice)
                claims_subsets = [x[1] for x in enumerate(claims_subsets) if x[0] in choices]

            print('no claims', len(claims))
            print('no of subsets:', len(claims_subsets))
            for claims_subset in claims_subsets:
                claims_with_features = ' <EOC>￨S '.join(process_instance(claims_subset))
                claims_texts         = ' <EOC> '.join(list(map(lambda x: x['text'], claims_subset)))

                # Mask over the claims
                claims_tokens = claims_with_features.split(' ')
                claims_mask   = [1 if '-CT' in token else 0 for token in claims_tokens]
                resulted_mask.append({'logits': claims_mask, 'words': claims_tokens})

                resulted_data.append([claims_texts, claims_with_features, conclusion_text, conc_entity])

    return resulted_data, resulted_mask

def save_data(output_path, data):
    #mapping into summary files
    with open(output_path + '_plain.txt.src', 'w', encoding='utf-8') as file:
        for x in data:
            file.write(x[0] + '\n')

    with open(output_path + '_features.txt.src', 'w', encoding='utf-8') as file:
        for x in data:
            file.write(x[1] + '\n')

    with open(output_path + '.txt.target', 'w', encoding='utf-8') as file:
        for x in data:
            file.write(x[2] + '\n')

    with open(output_path + '_only_tgt.txt.target', 'w', encoding='utf-8') as file:
        for x in data:
            file.write(x[3] + '\n')

def save_mask(mask, path):
    with open(path, 'w') as file:
        for item in mask:
            file.write(json.dumps(item) + '\n')

def split_data(input_path, output_path):
    tagged_json_obj = json.load(open(input_path))


    debates = np.unique(list(map(lambda x: x['_debate_id'], tagged_json_obj)))

    print('total number of debates:', len(debates))

    train_debates = debates[0:450]
    dev_debates   = debates[450:517]
    test_debates  = debates[517:]

    print('train debates:', len(train_debates))
    print('valid debates:', len(dev_debates))
    print('test debates:', len(test_debates))

    train_instances = list(filter(lambda x: x['_debate_id'] in train_debates, tagged_json_obj))
    dev_instances   = list(filter(lambda x: x['_debate_id'] in dev_debates, tagged_json_obj))
    test_instances  = list(filter(lambda x: x['_debate_id'] in test_debates, tagged_json_obj))

    json.dump(train_instances, open(output_path + 'train.json', 'w'))
    json.dump(dev_instances, open(output_path + 'dev.json', 'w'))
    json.dump(test_instances, open(output_path + 'test.json', 'w'))

def generate_summ_files(input_path, output_path, k):
    data = json.load(open(input_path))
    
    if 'test' in input_path:
        data_instances, data_mask  = test_to_summary_file(data, k=k)
        data_instances_and_mask = list(zip(data_instances, data_mask))
        #np.random.shuffle(data_instances_and_mask)
        instances, mask = zip(*data_instances_and_mask)

        save_mask(mask, output_path + '.test_mask.json')
        save_data(output_path + '.test', instances)
    else:
        output_path = (output_path + 'dev') if ('dev' in input_path) or ('valid' in input_path) else (output_path + 'train')
        
        if ('dev' in input_path) or ('valid' in input_path):
            data_instances, _  = test_to_summary_file(data, k=k)
        else:
            data_instances, _ = train_to_summary_file(data, k=k, only_top_claims=False, max_choice=10)
            np.random.shuffle(data_instances)
            
        save_data(output_path, data_instances)

def tag_instance_using_flair(target_tagger, ner_tagger, pos_tagger, instance):
    print('processing:', instance[0])
    instance = instance[1]

    conclusion = instance['_claim']
    claims = list(instance['_argument_sentences'].values())

    #predict targets...
    conclusion_sent = Sentence(conclusion)
    claims_sents    = list(map(lambda x : Sentence(x), claims))
    
    target_tagger.predict([conclusion_sent] + claims_sents)
    ner_tagger.predict([conclusion_sent] + claims_sents)
    pos_tagger.predict([conclusion_sent] + claims_sents)
  

    tagged_claims = []
    for i, c in enumerate(claims_sents):
        tagged_claims.append({
            'text': claims[i],
            'pos' : c.to_dict(tag_type='pos')['entities'],
            'named_entities' : c.to_dict(tag_type='ner')['entities'],
            'targets' : c.to_dict(tag_type='ct')['entities']
        })

    return {
        '_debate_id': instance['_debate_id'],
        'conclusion': {'text': conclusion, 
                       'pos': conclusion_sent.to_dict(tag_type='pos')['entities'], 
                       'named_entities': conclusion_sent.to_dict(tag_type='ner')['entities'],
                       'targets': conclusion_sent.to_dict(tag_type='ct')['entities']
                    },
        'claims': tagged_claims
    }

def tag_content_using_flair(claim_target_tagger_path, input_path, output_path):
    data = json.load(open(input_path))

    target_tagger = SequenceTagger.load_from_file(claim_target_tagger_path)
    ner_tagger    = SequenceTagger.load('ner')
    pos_tagger    = SequenceTagger.load('pos')

    tagged_content = list(map(lambda x: tag_instance_using_flair(target_tagger, ner_tagger, pos_tagger, x), enumerate(data)))
    open(output_path + '.json', 'w').write(json.dumps(tagged_content))

def tag_instance_using_stanford(nlp_parser, instance):
    print('processing: ', instance[0])
    instance = instance[1]

    conclusion = instance['_claim']
    claims = list(instance['_argument_sentences'].values())

    #predict pos, ne,....
    tagged_items = nlp_parser.sentences_to_tags([conclusion] + claims)
    named_entities = nlp_parser.extract_named_entities([conclusion] + claims)


    conclusion_tags = tagged_items[0]
    claims_tags     = tagged_items[1:]

    conclusion_nes = named_entities[0]
    claims_nes     = named_entities[1:]

    claims = list(map(lambda x: {'text': x[0], 
                                 'tags': x[1], 
                                 'named_entities': x[2]
                            }, zip(claims, claims_tags, claims_nes)))

    print('Done...', flush=True)

    return {
        '_debate_id': instance['_debate_id'],
        'conclusion': {'text': conclusion, 
                       'tags': conclusion_tags, 
                       'named_entities': conclusion_nes
                    },
        'claims': claims
    }

def tag_content_using_stanford(input_path, output_path):
    import multiprocessing as mp

    data = json.load(open(input_path))

    nlp_parser = NlpParser()
    
    tag_instance_func = partial(tag_instance, nlp_parser)
    with mp.Pool() as pool:
        tagged_content = pool.map(tag_instance_func, enumerate(data))
        open(output_path + '.json', 'w').write(json.dumps(tagged_content))
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('task', type=str)
    parser.add_argument('--input_path', dest='input_path', type=str)
    parser.add_argument('--output_path', dest='output_path',  type=str)
    parser.add_argument('--tagger_path', dest='tagger_path',  type=str)
    parser.add_argument('--k', dest='k',  type=int)

    args = parser.parse_args()

    if args.task == 'split':
        split_data(args.input_path, args.output_path)
    if args.task == 'summ':
        generate_summ_files(args.input_path, args.output_path, k=args.k)
    if args.task == 'tag':
        tag_content_using_flair(args.tagger_path, args.input_path, args.output_path)
