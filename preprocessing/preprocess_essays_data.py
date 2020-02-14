import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)

import argparse
import json
import glob
import csv
import lxml.etree as et
import nltk
from sequence_modeling import utils

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def data2text(instances):
    srcs = []
    trgs = []
    for instance in instances:
        main_claim = instance['c']['text']
        main_claim = ' '.join(nltk.word_tokenize(main_claim))

        claims = ' '.join(list(map(lambda x: x['text'], instance['cs'])))
        claims = ' '.join(nltk.word_tokenize(claims))

        srcs.append(claims)
        trgs.append(main_claim)

    return srcs, trgs


def data2xml(d, name='data'):
    r = et.Element(name)
    return et.tostring(buildxml(r, d), encoding='utf-8')

def buildxml(r, d):
    if isinstance(d, dict):
        for k, v in d.items():
            s = et.SubElement(r, k)
            buildxml(s, v)
    elif isinstance(d, tuple) or isinstance(d, list):
        for v in d:
            s = et.SubElement(r, 'i')
            buildxml(s, v)
    else:
        r.text = d
    return r

def find_sent(sents, subs):
    for s in sents:
        if subs in s:
            #print(subs)
            #print(s)
            return ' '.join(nltk.word_tokenize(s))

    return ' '.join(nltk.word_tokenize(subs))

def extract_claims(ann_path, only_mainclaim=False):
    result = []
    content = open(ann_path, encoding='utf-8').readlines()

    essay_text = open(ann_path.replace('.ann', '.txt'), encoding='utf-8').read()
    essays_sents = nltk.sent_tokenize(essay_text)
    essays_sents = list(map(lambda x: x, essays_sents))

    #List of all annotations
    Ts = list(filter(lambda x: x.startswith('T'), content))
    Ts = list(map(lambda x: x.split('\t'), Ts))
    Ts = dict(map(lambda x: (x[0], (x[2], x[1])), Ts))
    
    #List of all relations
    Rs = list(filter(lambda x: x.startswith('R'), content))
    Rs = list(map(lambda r: r.split('\t')[1].split(' '), Rs))
    Rs = list(map(lambda r: (r[1].split(':')[1], r[2].split(':')[1], r[0]), Rs))

    As = list(filter(lambda x: x.startswith('A'), content))
    As = list(map(lambda r: r.split('\t')[1].split(' '), As))
    As = dict(map(lambda r: (r[1].strip(), r[2].strip()), As))


    #take only one major claim
    mc  = list(filter(lambda x: x[1][1].startswith('MajorClaim'), Ts.items()))[0]
    cs = list(filter(lambda x: x[1][1].startswith('Claim'), Ts.items()))
    cs = list(map(lambda v: {'text': find_sent(essays_sents, Ts[v[0]][0].strip()), 'rel': 'supports' if As[v[0]] == 'For' else 'against'}, cs))

    if cs != []:
       result.append({'c': {'text': find_sent(essays_sents, mc[1][0]), 'lvl': mc[1][1]}, 'cs': cs})

    if only_mainclaim:
        #Just to extract one main claim from each essay
        return result


    x = 0
    #Loop until we find one main claim then exit
    for k in Ts:
        v = Ts[k]

        if only_mainclaim and 'MajorClaim' not in v[1]:
            continue

        c = {'text': find_sent(essays_sents, v[0]), 'lvl': v[1]}
        #attach supporitng/attacking claims
        x+=1
        cs = []
        for r in Rs:
            if r[1] == k:
                cs.append({'text': find_sent(essays_sents, Ts[r[0]][0]), 'rel': r[2]})

        # for a in As:
        #     cs.append({'text': Ts[a[0]][0], 'rel': a[2]})

        if cs != []: # A claim that doesnt have any support/attack
            result.append({'c': c, 'cs': cs})

        
    return result

def check_overlap(all_claims):
    for item in all_claims:
        print('============ conclusion ================')
        print(item['c']['text'])
        print('--- claims: ----')
        for x in item['cs']:
            print(x['text'])
            if item['c']['text'] in x['text']:
                print('overlapp...')

def tag_instance_using_flair(target_tagger, ner_tagger, pos_tagger, instance):
    inst_id = instance[0]
    print('processing:', inst_id)
    instance = instance[1]

    conclusion = instance[1]
    claims = instance[0]

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
        'id': inst_id,
        'conclusion': {'text': conclusion, 
                       'pos': conclusion_sent.to_dict(tag_type='pos')['entities'], 
                       'named_entities': conclusion_sent.to_dict(tag_type='ner')['entities'],
                       'targets': conclusion_sent.to_dict(tag_type='ct')['entities']
                    },
        'claims': tagged_claims
    }

def tag_student_essays(tagger_path, corpus_path, output_path):
    target_tagger = SequenceTagger.load_from_file(tagger_path)
    ner_tagger    = SequenceTagger.load('ner')
    pos_tagger    = SequenceTagger.load('pos')

    data = utils.load_essays_data(corpus_path, include_only_pro_claims=True)
    
    print(data[0])
    tagged_samples = list(map(lambda x: tag_instance_using_flair(target_tagger, ner_tagger, pos_tagger, x), enumerate(data)))
    open(output_path, 'w').write(json.dumps(tagged_samples))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('task', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--tagger_path', type=str)
    parser.add_argument('output_as', type=str)    


    args = parser.parse_args()

    if args.task == 'extract':
        input_path = args.data_path
        ann_path   = input_path + '/*.ann'

        all_claims = []
        for p in glob.glob(ann_path):
            all_claims = all_claims + extract_claims(p, False)


        if args.output_as == 'XML':
            open(args.output_path, 'w').write(data2xml(all_claims).decode('utf-8'))
        else:
            srcs, trgs = data2text(all_claims)
            open(args.output_path + '.txt.src', 'w').write("\n".join(srcs))
            open(args.output_path + '.txt.target', 'w').write("\n".join(trgs))

    if args.task == 'tag':
        tag_student_essays(args.tagger_path, args.args.data_path, args.output_path)