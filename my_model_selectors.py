import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        num_features = 4

        score_bic = float('-inf')
        model_bic = None
        logL_array = []

        for component_n in range(self.min_n_components, self.max_n_components + 1):
            for x_train, len_train in self.sequences:

                try:
                    p_param = component_n * component_n + 2 * component_n * len(self.X[0]) - 1
                    model = GaussianHMM(n_components=component_n, n_iter=1000).fit(x_train, len_train)
                    logL = model.score(x_train, len_train)
                    running_score = -2 * logL + p_param * math.log(len(self.sequences))
                except:
                    pass

            if running_score > score_bic:
                score_bic = running_score
                model_bic = model

        return model_bic

        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        score_cv = float('-inf')
        model_cv = None
        logL_array = []
        #kf = KFold(n_splits=min(3, len(self.lengths)))
        #num_hidstates = self.max_n_components - self.min_n_components

        for component_n in range(self.min_n_components, self.max_n_components + 1):
            kf = KFold(n_splits=min(3, len(self.lengths)))

            for train_n, test_n in kf.split(self.sequences):
                try:
                    #x1, len1 = train_n.get_word_Xlengths(self.words)
                    x_train, len_train = combine_sequences(train_n, self.sequences)
                    x_test, len_test = combine_sequences(test_n, self.sequences)

                    model = GaussianHMM(n_components=component_n, n_iter=1000).fit(x_train, len_train)
                    logL = model.score(x_test, len_test)
                    logL_array.append(logL)
                except:
                    pass

            mean_score = np.mean(logL_array)

            if mean_score > score_cv:
                score_cv = mean_score
                model_cv = model

        return model_cv

        #raise NotImplementedError
