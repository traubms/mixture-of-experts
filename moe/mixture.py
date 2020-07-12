from abc import ABC, abstractmethod

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error


class MixtureOfExpertsMixin(BaseEstimator, ABC):
    """ mixture of experts mixin to provide framework for EM fitting of MoE

    Args:
         experts: list of classifiers
         gate: classifier to pick between experts
         max_iter: int, max EM iterations; default=50

    Attributes:
        num_experts_: int, number of experts
    """

    def __init__(self, experts, gate, tol=1e-4, max_iter=100):
        self.experts = experts
        self.gate = gate
        self.tol = tol
        self.max_iter = max_iter

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        """ data log-likelihood

        For experts j,
            score = \sum_i w_i log( \sum_j g_j(i) P_j[y_i | x_i] )

        Args:
            X: n x k data matrix
            y: n x 1 vector of targets
            sample_weight: n x 1 array of weights; default=None, equal weight

        Returns:
            float
        """
        pass

    @abstractmethod
    def _estep(self, X, y):
        """ determine soft data assignments to experts

        Args:
            X: n x k data matrix
            y: n x 1 targets

        Returns:
            n x m matrix of assignments of data to experts
        """
        pass

    def fit(self, X, y):
        self.num_experts_ = len(self.experts)
        self._init(X, y)
        self.obj_vals = []
        while True:
            expert_weights = self._estep(X, y)
            obj_val = self._mstep(X, y, expert_weights)
            self.obj_vals.append(obj_val)

            if len(self.obj_vals) < 2:
                continue

            if abs(self.obj_vals[-2] - self.obj_vals[-1]) <= self.tol or len(self.obj_vals) >= self.max_iter:
                break
        return self

    def _mstep(self, X, y, expert_weights):
        """ refit experts and gate given soft data assignments to experts

        Args:
            X: n x k data matrix
            y: n x 1 targets
            expert_weights: n x m matrix of assignments of data to experts

        Returns:
            float, new obj function value
        """
        self.gate.fit(X, expert_weights)

        for expert_index in range(self.num_experts_):
            expert_sample_weight = expert_weights[:, expert_index]
            self.experts[expert_index].fit(X, y, sample_weight=expert_sample_weight)

        return self.score(X, y)

    def _init(self, X, y):
        """ initialize experts and gate on random data with same mean / std

        Args:
            X: n x k data matrix
            y: n x 1 vector of targets
        """

        X_center = np.mean(X, axis=0)
        X_scale = np.std(X, axis=0)
        batch = min(len(y), 100)
        X_init = np.random.multivariate_normal(X_center, np.diag(X_scale), size=batch * self.num_experts_)

        for i, expert in enumerate(self.experts):
            expert.fit(X_init[i * batch:(i + 1) * batch], y[:batch])
        random_init = np.random.rand(X.shape[0], self.num_experts_)
        random_init = random_init / random_init.sum(axis=1)[:, None]
        self.gate.fit(X, random_init)


class MixtureOfExpertsClassifier(MixtureOfExpertsMixin, ClassifierMixin):
    """ mixture of experts classifier

    Args:
         experts: list of classifiers
         gate: classifier to pick between experts
         max_iter: int, max EM iterations; default=50
    """

    def _estep(self, X, y):

        """
        description: finds the contribution of each expert to final prediction
        input:	X - data matrix
                y - label matrix
        output: N x M matrix of feature weights for each point for each expert
        """

        weighted_expert_accuracy = self._weighted_expert_accuracy(X, y)
        feature_weights = self._get_expert_weights(weighted_expert_accuracy)

        return feature_weights

    def score(self, X, y, sample_weight=None):
        """ data log-likelihood

        For experts j,

            score = \sum_i w_i log( \sum_j g_j(i) P_j[y_i | x_i] )

        Args:
            X: n x k data matrix
            y: n x 1 vector of targets
            sample_weight: n x 1 array of weights; default=None, equal weight

        Returns:
            float
        """

        weighted_expert_accuracy = self._weighted_expert_accuracy(X, y)
        expert_weights = self._get_expert_weights(weighted_expert_accuracy)
        log_prob = np.multiply(np.log(weighted_expert_accuracy), expert_weights)
        if sample_weight != None:
            log_prob = np.multiply(log_prob, sample_weight)

        return np.sum(log_prob)

    def fit(self, X, y):
        self.num_classes_ = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))
        return super().fit(X, y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """ probability X belongs to each class

        Args:
            X: n x k data matrix

        Returns:
            n x d matrix of probabilities
        """
        expert_predictions = self.predict_proba_experts(X)
        gate_proba = self.gate.predict_proba(X)
        gate_proba_big = np.empty((X.shape[0], self.num_classes_, self.num_experts_))
        for k in range(self.num_classes_):
            gate_proba_big[:, k, :] = gate_proba

        gated_expert_accruacy = np.multiply(expert_predictions, gate_proba_big)
        return np.sum(gated_expert_accruacy.reshape(X.shape[0], self.num_classes_, self.num_experts_), axis=2)

    def predict_proba_experts(self, X):
        """ probability X belongs to each class according to each expert

        Args:
            X: n x k data matrix

        Returns:
            n x d x m matrix of probabilities for each expert
        """
        predictions = np.zeros((X.shape[0], self.num_classes_, self.num_experts_))
        for i, expert in enumerate(self.experts):
            predictions[:, :, i] = expert.predict_proba(X)
        return predictions

    def _weighted_expert_accuracy(self, X, y):

        """
        description: returns matrix A_ij = g_j (x_i) * P(y_i | x_i, j)
        input:	X - input matrix
                y - output matrix
        output: gates expert predictions in N x M matrix as described above
        """
        expert_predictions = self.predict_proba_experts(X)
        expert_accuracy = np.multiply(expert_predictions, y[:, :, np.newaxis])
        # expert_accuracy = expert_predictions
        # gap = 0
        gate_proba = self.gate.predict_proba(X)
        gate_proba_big = np.empty((X.shape[0], self.num_classes_, self.num_experts_))
        for k in range(self.num_classes_):
            gate_proba_big[:, k, :] = gate_proba

        gated_expert_accruacy = np.multiply(expert_accuracy, gate_proba_big)
        norm_weights = gated_expert_accruacy.reshape(X.shape[0], self.num_classes_, self.num_experts_)

        # gated_expert_accruacy = expert_accuracy
        return np.sum(norm_weights, axis=1)

    def _get_expert_weights(self, weighted_expert_accuracy):
        return np.divide(weighted_expert_accuracy, np.sum(weighted_expert_accuracy, axis=1)[:, np.newaxis])


class MixtureOfExpertsRegressor(MixtureOfExpertsMixin, RegressorMixin):
    """ mixture of experts classifier

    Args:
         experts: list of classifiers
         gate: classifier to pick between experts
         max_iter: int, max EM iterations; default=50

    Attributes:
        expert_scale_: fitted residual std deviation for each expert
        count_prior_: prior of weight to give to std_prior in computing expert scale
        scale_prior_: prior for expert scale
    """

    def score(self, X, y, sample_weight=None):
        return RegressorMixin.score(self, X, y, sample_weight=sample_weight)

    def fit(self, X, y):
        self.count_prior_ = len(X) / len(self.experts) / 100
        self.scale_prior_ = np.std(y)
        self.expert_scale_ = np.ones(len(self.experts)) * self.scale_prior_
        self._expert_preds = None
        return super().fit(X, y)

    def predict(self, X):
        yhat_experts = self.predict_experts(X)
        probs = self.gate.predict_proba(X)
        return (yhat_experts * probs).sum(axis=1)

    def predict_experts(self, X):
        yhat_experts = np.zeros((len(X), self.num_experts_))
        for i, expert in enumerate(self.experts):
            yhat_experts[:, i] = expert.predict(X)
        return yhat_experts

    def _estep(self, X, y):
        """ determine soft data assignments to experts

        Args:
            X: n x k data matrix
            y: n x 1 targets

        Returns:
            n x m matrix of assignments of data to experts
        """
        log_scale = np.log(self.expert_scale_)
        log_likelys = np.zeros((len(X), self.num_experts_))
        for i, expert in enumerate(self.experts):
            if self._expert_preds is None:
                yhat_expert = expert.predict(X)
            else:
                yhat_expert = self._expert_preds[:, i]
            scale = self.expert_scale_[i]
            log_likelys[:, i] = -.5 / scale ** 2 * np.square(y - yhat_expert) - log_scale[i]

        prior_log_probs = self.gate.predict_log_proba(X)

        log_posterior = log_likelys + prior_log_probs
        log_sum_posterior = logsumexp(log_posterior, axis=1)
        expert_weights = np.exp(log_posterior - log_sum_posterior[:, None])
        return expert_weights

    def _mstep(self, X, y, expert_weights):
        out = super()._mstep(X, y, expert_weights)

        # fit expert scale
        scales = np.zeros(self.num_experts_)
        counts = np.zeros(self.num_experts_)
        self._expert_preds = np.zeros((len(X), self.num_experts_))
        for i, expert in enumerate(self.experts):
            yhat_expert = expert.predict(X)
            scales[i] = mean_squared_error(y, yhat_expert, expert_weights[:, i])
            counts[i] = expert_weights[:, i].sum()
            self._expert_preds[:, i] = yhat_expert

        self.expert_scale_ = np.sqrt(
            (scales ** 2 * counts + self.scale_prior_ ** 2 * self.count_prior_) / (counts + 2 * self.count_prior_))

        return out
