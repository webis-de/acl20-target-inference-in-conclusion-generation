import os
import sys
import argparse

nb_dir = os.path.split(os.getcwd())[0]
sys.path.append(nb_dir)

import json
import numpy as np
import pandas as pd

from target_inference.conclusion_target_modeling import *
from target_inference.evaluation_measures import *


def best_conclusions_targets(data_path):
    conclusions = []
    dataset = json.load(open(data_path, 'r'))

    overlaps = 0
    for sample in dataset:
        conclusion_target_text = sample['conclusion']['targets'][0]['text'] if len(sample['conclusion']['targets']) > 0 else '<NONE>' ##sample['conclusion']['text']
        targets = [claim['targets'][0]['text'] if len(claim['targets']) > 0 else ' ' for claim in  sample['claims']]
        for target in targets:
            if overlap_case(target, conclusion_target_text):
                print(target)
                print(conclusion_target_text)
                print('=================')
                overlaps +=1

    print('overlaps perc:', overlaps/len(dataset))
    print('total cases:', len(dataset))


def generate_conclusions_targets(args):
    dataset = json.load(open(args.data_path, 'r'))
    
    if args.model_type == 'hyp':
        print('ranking and embedding ...')
        model = HyperdModeling(load_concepts=False, embedding_method=args.embedding_method)
        model.fit_on_data([])
        targets = [model.get_target(x['claims'], simple_avg=simple_avg) for x in dataset]

    if args.model_type == 'target_embedding_simple':
        print('generate conclusions based on simple averaging ...')
        model = TargetSpaceModeling(embedding_method=args.embedding_method, src_target_space_path=args.src_target_file)
        targets = [model.get_target(x['claims'], simple_avg=True) for x in dataset]

    if args.model_type == 'target_embedding_learned':
        print('generate conclusions based on learned target space ...')
        model = TargetSpaceModeling(siamese_model_path=args.siamese_models[0],  embedding_method=args.embedding_method, src_target_space_path=args.src_target_file)
        targets = [model.get_target(x['claims'], simple_avg=False) for x in dataset]

    if args.model_type == 'ranked':
        print('generate conclusions based ranked targets ...')
        model = RankedTargetModel()
        targets = [model.get_target(x['claims']) for x in dataset]

    
    save_results(args.output_path, targets)
    print('evaluate accuracy:')
    acc = eval_acc(args.gt_path, args.output_path)
    meteor = eval_meteor(args.gt_path, args.output_path)
    meteor = eval_meteor(args.gt_path, args.output_path)
    bleu   = eval_bleu(args.gt_path, args.output_path)
    print('Accuracy:', acc, ' Meteor:', meteor, ' Bleu:', bleu)


def choose_best_model(args):
    test_dataset = json.load(open(args.data_path, 'r'))

    for siamese_model_path in args.siamese_models:
        #Build the approach; hyp or only target embedding
        if args.model_type == 'hyp':
            print('ranking and embedding ...')
            model = HyperdModeling(load_concepts=False, siamese_model_path=siamese_model_path, embedding_method=args.embedding_method, src_target_space_path=args.src_target_file)
        if args.model_type == 'target_embedding_learned':
            print('generate conclusions based on learned target space ...')
            model = TargetSpaceModeling(load_concepts=False, siamese_model_path=siamese_model_path,  embedding_method=args.embedding_method, src_target_space_path=args.src_target_file)
        
        targets = [model.get_target(x['claims'], simple_avg=False) for x in test_dataset]

        model_epoch = siamese_model_path.split('/')[-1].replace('.pth', '')
        preds_output_path = args.output_path + model_epoch + '_preds.out'
        save_results(preds_output_path, targets)
        acc = eval_acc(args.gt_path, preds_output_path)
        print('Accuracy of {} is {}:'.format(model_epoch, acc))
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('task', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--src_target_file', type=str)
    parser.add_argument('--ranking_model', type=str)
    parser.add_argument('--siamese_models', nargs='+')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--embedding_method', type=str, default='glove')


    args = parser.parse_args()

    if args.task == 'choose_best_model':
        choose_best_model(args)
    elif args.task == 'generate_preds':
        generate_conclusions_targets(args)


#python targets_space_experiment.py --data_path ../data/idebate/test_scored.json --train_path ../data/idebate/train_scored.json --output_path ./output_results/ --gt_path ../data/idebate/results/test_files/top_5.test_only_tgt.txt.target