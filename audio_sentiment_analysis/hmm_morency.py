'''
Implements an HMM classifier as described in Morency, et al.
(http://dl.acm.org/citation.cfm?id=2070509)

Author: Avery Wells
'''
import numpy as np
from sklearn.model_selection import  LeaveOneGroupOut
from hmmlearn import hmm
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')


def separate_classes(data_train, labels_train):
    '''
    Separates input data into different data sets based on class. Will
    produce positive and negative training sets.
    '''
    pos_index = np.where(labels_train == 1)
    neg_index = np.where(labels_train == 0)

    return data_train[pos_index[0]], data_train[neg_index[0]]


def classify_predictions(pos_model, neg_model, data_test):
    '''
    Assigns a label to a sample based upon which classifier gave it a
    higher probability.
    '''
    labels_pred = []
    for sample in data_test:
        labels_pred.append(0 if neg_model.score(sample) > pos_model.score(sample) else 1)

    return np.array(labels_pred)

def main():
    '''
    Performs training and classification of given data.
    '''
    data_prefix = '../../data/pre-extracted/'
    data = np.loadtxt(data_prefix + 'YouTube_acoustic.csv', delimiter=',')
    labels = np.loadtxt(data_prefix + 'YouTube_sentiment_label.csv', delimiter=',')
    ids = np.loadtxt(data_prefix + 'YouTube_subject_id.csv', delimiter=',')

    # group the data by speaker
    logo = LeaveOneGroupOut()

    # init the model with set hyperparameters
    components = 4
    mix = 4
    pos_model = hmm.GMMHMM(n_components=components, n_mix=mix, random_state=1)
    neg_model = hmm.GMMHMM(n_components=components, n_mix=mix, random_state=1)

    # do the training/testing
    precisions, recalls, f_scores = [], [], []
    for train_index, test_index in logo.split(data, labels, ids):
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # separate training data into two sets: one for each class
        pos_train, neg_train = separate_classes(data_train, labels_train)

        # train
        pos_model.fit(pos_train)
        neg_model.fit(neg_train)

        # test
        labels_pred = classify_predictions(pos_model, neg_model, data_test)

        # evaluate
        precision, recall, f_score, support = precision_recall_fscore_support(labels_test, labels_pred, average='weighted')
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)

    # output avg
    print('Avg F1: ' + str(np.mean(f_scores)))
    print('Avg Precision: ' + str(np.mean(precisions)))
    print('Avg Recall: ' + str(np.mean(recalls)))


if __name__ == '__main__':
    # Will be executed when module is run directly
    main()
