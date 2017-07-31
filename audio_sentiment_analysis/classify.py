'''
Classifies audio chunks as having a positive or negative sentiment. Can
select from several possible models or classify with all available.

Takes as input a feature CSV as output produced by process_raw_data.py
In general, the CSV format should:
- Have column headers
- The first column should be the call ID (assuming multiple chunks from same call)
- Last column contains a label of 0 or 1

Author: Avery Wells 2017
'''

import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import svm
from hmm_morency import HmmMorency
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_and_test(model, data_train, data_test, labels_train, labels_test):
    '''
    Fits the given model to the training data and calculates its classification
    accuracy on the test set. Returns classification accuracy.
    '''
    model.fit(data_train, labels_train)
    pred_labels = model.predict(data_test)
    chunk_scores = accuracy_score(labels_test, pred_labels)

    # assigns a single label to a whole call based on whether its chunks were
    # majority positive or negative
    pred_call_label = np.round(np.mean(pred_labels))
    if pred_call_label == labels_test[0]:
        call_score = 1.0
    else:
        call_score = 0.0

    pred.append(pred_call_label)
    true.append(labels_test[0])

    return chunk_scores, call_score


def sep_data_labels(feat_loc):
    '''
    Separates the data and labels from the input CSV feature file. Also
    removes all headers and saves the IDs from the first column to be used
    as group IDs for leave-one-group-out validation.
    Returns data, labels, and IDs
    '''
    feature_file = pd.read_csv(feat_loc)
    ids = feature_file.ix[:, 0].values
    labels = feature_file.ix[:, -1].values
    data = feature_file.ix[:, 1:-1].values

    return shuffle(data, labels, ids, random_state=10)


def score_stats(model, chunk_scores, overall_scores, out_file):
    '''
    Calculates basic statistics on a model's scores: max, min, avg, std dev.
    Writes stats to an output file.
    '''
    max_score = np.amax(chunk_scores)
    min_score = np.amin(chunk_scores)
    avg_score = np.mean(chunk_scores)
    std_score = np.std(chunk_scores)
    overall = np.mean(overall_scores)

    with open(out_file, 'a') as results:
        results.write(model + ': \n')
        results.write('Max: ' + str(max_score) + '\n')
        results.write('Min: ' + str(min_score) + '\n')
        results.write('Std: ' + str(std_score) + '\n')
        results.write('Chunk Avg: ' + str(avg_score) + '\n')
        results.write('Call Avg: ' + str(overall) + '\n\n')


def main(args, pipe=False):
    '''
    Checks passed arguments and performs requested actions.
    '''
    if not pipe:
        parser = argparse.ArgumentParser(description='Classify call segments as positive or negative.')
        parser.add_argument('-f', '--features', dest='feat_loc', required=True,
                            help='Path to CSV feature file.')
        parser.add_argument('-o', '--out', dest='out_loc', required=True,
                            help='Path to where classification summary should be saved.')
        parser.add_argument('--hmm', dest='hmm_flag', action='store_true',
                            help='Classify with a Hidden Markov Model.')
        parser.add_argument('--rf', dest='rf_flag', action='store_true',
                            help='Classify with a random forest.')
        parser.add_argument('--n_components', dest='n_components',
                            help='Number of components for the HMM.')
        parser.add_argument('--n_mix', dest='n_mix',
                            help='Number of Gaussian mixtures for the HMM.')
        parser.add_argument('--n_estimators', dest='n_estimators',
                            help='Number of tree estimators for the random forest.')
        args = parser.parse_args()

    if args.hmm_flag or args.rf_flag:
        # store scores from all runs to calc stats
        hmm_chunk_scores = []
        hmm_overall_scores = []
        rf_chunk_scores = []
        rf_overall_scores = []

        # split data for leave-one-group(call)-out validation
        data, labels, ids = sep_data_labels(args.feat_loc)
        logo = LeaveOneGroupOut()
        curr_split = 1
        num_splits = logo.get_n_splits(data, labels, ids)
        
        # loop through all cross validation folds
        for train_index, test_index in logo.split(data, labels, ids):
            print('Split ' + str(curr_split) + ' out of ' + str(num_splits))
            data_train, data_test = data[train_index], data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            # classify with the selected models
            if args.hmm_flag:
                if args.n_components:
                    n_components = int(args.n_components)
                else:
                    n_components = 2
                if args.n_mix:
                    n_mix = int(args.n_mix)
                else:
                    n_mix = 2
                hmm_model = HmmMorency(n_components=n_components, n_mix=n_mix)
                chunk_scores, call_score = train_and_test(hmm_model, data_train, data_test, labels_train, labels_test)
                hmm_chunk_scores.append(chunk_scores)
                hmm_overall_scores.append(call_score)

            if args.rf_flag:
                if args.n_estimators:
                    n_estimators = int(args.n_estimators)
                else:
                    n_estimators = 100
                rf_model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=10)
                chunk_scores, call_score = train_and_test(rf_model, data_train, data_test, labels_train, labels_test)
                rf_chunk_scores.append(chunk_scores)
                rf_overall_scores.append(call_score)

            curr_split += 1

        # evaluate the scores for all models
        out_file = os.path.join(args.out_loc, 'results.txt')
        if args.hmm_flag:
            score_stats('hmm, mix: ' + str(n_mix) + ' states: ' + str(n_components),
                        hmm_chunk_scores, hmm_overall_scores, out_file)
        if args.rf_flag:
            score_stats('random forest, estimators: ' + str(n_estimators),
                        rf_chunk_scores, rf_overall_scores, out_file)

    else:
        sys.exit('Must choose at least one classification method. (--hmm, --rf)')


if __name__ == '__main__':
    main(sys.argv[1:])
