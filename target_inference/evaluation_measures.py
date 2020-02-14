from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.corpus import stopwords

import random

from scipy import stats


def filter_no_target_cases(reference, prediction):
    #print(reference)
    non_indices = [i for i,e in enumerate(reference) if e == '<NONE>']
    new_reference  = [item for idx, item in enumerate(reference) if idx not in non_indices]
    new_prediction = [item for idx, item in enumerate(prediction) if idx not in non_indices]

    #print(non_indices)
    #print(new_reference[0:15])
    #print(new_prediction[0:15])
    return new_reference, new_prediction

def overlap_case(predicted_target, true_target, threshold=0.5):
    true_target = true_target.strip().lower().split(' ')
    predicted_target = predicted_target.strip().lower().split(' ')

    english_stopwords = stopwords.words('english')

    true_target      = set([t for t in true_target if t not in english_stopwords])
    predicted_target = set([t for t in predicted_target if t not in english_stopwords])

    shared_tokens = true_target.intersection(predicted_target)
    all_tokens    = true_target.union(predicted_target)
    return len(shared_tokens)/len(all_tokens) >= threshold

def eval_acc(ref_file, pred_file, threshold=0.4):
    reference  = open(ref_file, 'r').read().split('\n')[:-1]
    prediction = open(pred_file, 'r').read().split('\n')[:-1]

    reference, prediction = filter_no_target_cases(reference, prediction)

    reference  = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').strip(), reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').strip(), prediction))

    return acc(reference, prediction, threshold)

def acc(reference, prediction, threshold):
    matching_cases = 0
    for x in zip(reference, prediction):
        if overlap_case(x[0], x[1], threshold=threshold):
            # print(x[0])
            # print(x[1])
            # print('============')
            matching_cases +=1


    return matching_cases/len(reference)

def eval_meteor(ref_file, pred_file):
    reference  = open(ref_file, 'r').read().split('\n')[:-1]
    prediction = open(pred_file, 'r').read().split('\n')[:-1]

    reference, prediction = filter_no_target_cases(reference, prediction)

    reference  = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), prediction))

    meteor_scores = [single_meteor_score(inst[0], inst[1]) for inst in zip(reference, prediction)]

    return sum(meteor_scores)/len(meteor_scores)

def eval_bleu(ref_file, pred_file, weights=[0.5, 0.5]):
    reference  = open(ref_file, 'r').read().split('\n')[:-1]
    prediction = open(pred_file, 'r').read().split('\n')[:-1]

    reference, prediction = filter_no_target_cases(reference, prediction)
    
    reference  = list(map(lambda x: [x.replace('<SCONC>', '').replace('<ECONC>', '').lower().split(' ')], reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower().split(' '), prediction))

    chencherry = SmoothingFunction()

    if weights != None:
        score = bleu_score.corpus_bleu(reference, prediction, weights=weights, smoothing_function=chencherry.method1)
    else:
        score = bleu_score.corpus_bleu(reference, prediction,  smoothing_function=chencherry.method1)

    return score * 100

def target_eval(ref_file, pred_file, remove_tags=False, weights=None):
    from flair.data import Sentence
    from flair.models import SequenceTagger

    reference  = open(ref_file, 'r').read().split('\n')[:-1]
    prediction = open(pred_file, 'r').read().split('\n')[:-1]

    reference  = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', ''), reference))
    prediction = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', ''), prediction))

    reference, prediction = filter_no_target_cases(reference, prediction)

    #tagger = SequenceTagger.load_from_file('/workspace/webis20_data/claim-target-tagger/best-model.pt')
    tagger = SequenceTagger.load_from_file('../models/claim_target_tagger.pt')

    b_score  = 0
    semantic_score = 0

    chencherry = SmoothingFunction()
    skipped_count = 0
    for pair in zip(reference, prediction):
        ref_sent, pred_sent  = tagger.predict([Sentence(pair[0]), Sentence(pair[1])])

        ref_targets  = [entity['text'] for entity in ref_sent.to_dict(tag_type='ct')['entities']]
        pred_targets = [entity['text'] for entity in pred_sent.to_dict(tag_type='ct')['entities']]

        print(ref_targets)
        print(pred_targets)
        print('=======')
        if len(ref_targets) == 0 or len(pred_targets) == 0:
            skipped_count += 1
            continue

        best_Score = max([ bleu_score.sentence_bleu([x], y, weights, smoothing_function=chencherry.method1) for x in ref_targets for y in pred_targets])
        b_score += best_Score

    print('Evaluated ', len(reference) - skipped_count, ' of total ', len(reference))
    return b_score/(len(reference) - skipped_count)

def compute_significance(ref_file, pred1_file, pred2_file, weights=None):
    reference   = open(ref_file, 'r').read().split('\n')[:-1]
    prediction1 = open(pred1_file, 'r').read().split('\n')[:-1]
    prediction2 = open(pred2_file, 'r').read().split('\n')[:-1]

    #Filter out cases with <NONE> as the target
    non_indices = [i for i,e in enumerate(reference) if e == '<NONE>']
    reference   = [item for idx, item in enumerate(reference) if idx not in non_indices]
    prediction1 = [item for idx, item in enumerate(prediction1) if idx not in non_indices]
    prediction2 = [item for idx, item in enumerate(prediction2) if idx not in non_indices]

    reference   = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), reference))
    prediction1 = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), prediction1))
    prediction2 = list(map(lambda x: x.replace('<SCONC>', '').replace('<ECONC>', '').lower(), prediction2))


    population = list(zip(reference, prediction1, prediction2))
    #random.shuffle(population)

    bleu_scores1 = []
    bleu_scores2 = []

    acc_scores1 = []
    acc_scores2 = []

    meteor_scores1 = []
    meteor_scores2 = []
    
    chencherry = SmoothingFunction()
    n=10
    samples = [population[i * n:(i + 1) * n] for i in range((len(population) + n - 1) // n )]  
    for sample in samples:
        #sample = population[i * 50 : (i+1) * 50]
        reference, prediction1, prediction2 = zip(*sample)
            

        acc_score1 = acc(reference, prediction1, 0.5)
        acc_score2 = acc(reference, prediction2, 0.5)

        meteor_scores = [single_meteor_score(inst[0], inst[1]) for inst in zip(reference, prediction1)]
        meteor_score1 = sum(meteor_scores)/len(meteor_scores)
        meteor_scores = [single_meteor_score(inst[0], inst[1]) for inst in zip(reference, prediction2)]
        meteor_score2 = sum(meteor_scores)/len(meteor_scores)
        
        reference   = [[x.split(' ')] for x in reference]
        prediction1 = [x.split(' ') for x in prediction1]
        prediction2 = [x.split(' ') for x in prediction2]

        bleu_score1 = bleu_score.corpus_bleu(reference, prediction1, weights=[0.5, 0.5], smoothing_function=chencherry.method1)
        bleu_score2 = bleu_score.corpus_bleu(reference, prediction2, weights=[0.5, 0.5], smoothing_function=chencherry.method1)


        bleu_scores1.append(bleu_score1*100)
        bleu_scores2.append(bleu_score2*100)

        acc_scores1.append(acc_score1*100)
        acc_scores2.append(acc_score2*100)

        meteor_scores1.append(meteor_score1*100)
        meteor_scores2.append(meteor_score2*100)

    bleu_ttest = stats.ttest_rel(bleu_scores1, bleu_scores2)
    acc_ttest = stats.ttest_rel(acc_scores1, acc_scores2)
    meteor_ttest = stats.ttest_rel(meteor_scores1, meteor_scores2)
    

    return [meteor_ttest, bleu_ttest, acc_ttest]