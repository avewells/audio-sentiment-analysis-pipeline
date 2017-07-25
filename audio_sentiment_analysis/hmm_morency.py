'''
Implements an HMM classifier as described in Morency, et al.
(http://dl.acm.org/citation.cfm?id=2070509)

Author: Avery Wells
'''

from hmmlearn import hmm
import numpy as np

class HmmMorency:
    '''
    HMM for two-class sentiment classification. A separate HMM is trained for each class.
    '''
    def __init__(self, n_components=4, n_mix=4, random_state=10):
        def build_hmm(n_components, n_mix):
            '''
            Create models with passed parameters.
            '''
            model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, random_state=random_state)
            return model
        self.hmm_0 = build_hmm(n_components, n_mix)
        self.hmm_1 = build_hmm(n_components, n_mix)

    def fit(self, x_train, y_train):
        '''
        Trains an HMM for each class. Follows sklearn style.
        '''
        x_neg = x_train[y_train == 0, :]
        x_pos = x_train[y_train == 1, :]

        self.hmm_0.fit(x_neg)
        self.hmm_1.fit(x_pos)

    def predict(self, x_test):
        '''
        Predicts labels on an unseen test set. Label from HMM with highest
        score is chosen.
        '''
        res = []
        for x in x_test:
            res.append(0 if self.hmm_0.score(x) > self.hmm_1.score(x) else 1)
        return np.array(res)
