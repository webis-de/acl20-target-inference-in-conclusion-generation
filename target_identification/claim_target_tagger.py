import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)


from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
from sequence_modeling import utils
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import nltk
import json
import argparse

def train_tagger(data_path, model_path):
    tag_type='ct'
    # define columns
    columns = {0: 'text', 1: 'pos', 2: 'ct'}
    # retrieve corpus using column format, data folder and the names of the train, dev and test files
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_path, columns, train_file='train.tsv', test_file='test.tsv')


    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),
        CharLMEmbeddings('news-forward'),
        CharLMEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


    tag_dictionary = corpus.make_tag_dictionary(tag_type='ct')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)



    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    # 7. start training
    trainer.train(model_path, learning_rate=0.1, mini_batch_size=16, max_epochs=30)


def tag_student_essays(tagger, corpus_path, output_path):
    data = utils.load_essays_data(corpus_path, include_only_pro_claims=False)
    
    tagged_samples = []
    for sample in data:
        conclusion = sample[1]
        claims = sample[0]

        tagged_main_claim = Sentence(conclusion)
        tagger.predict(tagged_main_claim)
        
        claims_features, tagged_claims = text_to_features(tagger, sample[0])

        tagged_samples.append({
            'conclusion': tagged_main_claim.to_dict(tag_type='ct'),
            'conclusion_features': [token.text+ u"\uFFE8" + token.tags['ct'].value for token in tagged_main_claim.tokens],
            'claims' : tagged_claims,
            'claims_text': ' '.join(claims),
            'claims_features': claims_features
        })
        
    open(output_path, 'w').write(json.dumps(tagged_samples))

def claims_to_mask(claims):
    logits = []
    words  = []
    for claim in claims:
        logits += [[1 - token.tags['ct'].score, token.tags['ct'].score] if token.tags['ct'].value == 'O' else [token.tags['ct'].score, 1 - token.tags['ct'].score]
                    for token in claim.tokens]
        words  += [token.text for token in claim.tokens]
    
    return {'logits': logits, 'words': words}

def text_to_features(tagger, sents):
    if type(sents) != list:
        sents = nltk.sent_tokenize(sents)
    
    tokens = []
    tagged_sents = []
    for sent in sents:
        sent = Sentence(sent)
        tagger.predict(sent)
        tokens += [token.text+ u"\uFFE8" + token.tags['ct'].value for token in sent.tokens]

        tagged_sents.append(sent.to_dict(tag_type='ct'))

    return ' '.join(tokens), tagged_sents

def json_to_summary_input(json_path, output_path, src_field='claims_features', skip_claims_without_target=False, generate_only_con_target=False):
    data = json.load(open(json_path, 'r'))

    src_txt = []
    tgt_txt = []

    for item in data:
        if skip_claims_without_target:
            if len(item['conclusion']['entities']) == 0:
                print('skipping instance..')
                continue

        src_txt.append(item[src_field].replace('|', u"\uFFE8"))

        if type(item['conclusion_features']) == list:
            tgt = ' '.join(item['conclusion_features']).replace('|', u"\uFFE8")
            if generate_only_con_target:
                tgt = ' '.join(list(filter(lambda x: 'B-C' in x or 'I-C' in x, tgt.split(' '))))
            
            tgt_txt.append((tgt).replace('|', u"\uFFE8"))
        else:
            tgt = item['conclusion_features'].replace('|', u"\uFFE8")
            if generate_only_con_target:
                tgt = ' '.join(list(filter(lambda x: 'B-C' in x or 'I-C' in x, tgt.split(' '))))

            tgt_txt.append(item['conclusion_features'].replace('|', u"\uFFE8"))

    with open(output_path + '.txt.src', 'w', encoding='utf-8') as file:
        for x in src_txt:
            file.write(x + '\n')

    with open(output_path + '.txt.target', 'w', encoding='utf-8') as file:
        for x in tgt_txt:
            file.write(x + '\n')

def tag_idebate_corpus(tagger, corpus_path, output_path):
    nltk.download('punkt')
    data = json.loads(open(corpus_path, 'r').read())
    
    def sample_features(sents):
        tokens = []
        if type(sents) == list :
            tokens = []
            for sent in sents:
                tokens += [token.text+ u"\uFFE8" + token.tags['ct'].value for token in sent.tokens]
            return ' '.join(tokens)
        else:
            return ' '.join([token.text+ u"\uFFE8" + token.tags['ct'].value for token in sents.tokens])


    tagged_samples = []

    for sample in data:
        conclusion = Sentence(sample['_claim'])
        claims = list(map(lambda x : Sentence(x), sample['_argument_sentences'].values()))

        tagged_items = tagger.predict([conclusion] + claims)

        tagged_conclusion = tagged_items[0]
        tagged_claims     = tagged_items[1:]

        

        conc_features = sample_features(conclusion)
        claims_features = sample_features(claims)
        
        tagged_samples.append({
            '_debate_id': sample['_debate_id'],
            'conclusion': tagged_conclusion.to_dict(tag_type='ct'),
            'claims': [x.to_dict(tag_type='ct') for x in tagged_claims],
            'conclusion_features': conc_features,
            'claims_features': claims_features
        })
        
    open(output_path + '.json', 'w').write(json.dumps(tagged_samples)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('task', type=str)
    parser.add_argument('--tag_corpus', dest='tag_corpus', type=str)
    parser.add_argument('--input_path', dest='input_path', type=str)
    parser.add_argument('--output_path', dest='output_path', type=str)

    args = parser.parse_args()

    if args.task == 'train':
        train_tagger('./data/ibm_cs', args.output_path)
    
    elif args.task == 'annotate':
        tagger = SequenceTagger.load_from_file(args.input_path)

        if args.tag_corpus == 'idebate':
            tag_idebate_corpus(tagger, args.input_path, args.output_path)
        else:
            tag_student_essays(tagger, args.input_path, args.output_path)

    elif args.task == 'summ_files':
        json_to_summary_input(args.input_path, args.output_path, src_field='claims_features', skip_claims_without_target=True, generate_only_con_target=True)