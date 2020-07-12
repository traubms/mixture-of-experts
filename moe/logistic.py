"""
Need to strip down sklearn LogisticRegression wrappers to support soft labels
"""

import numbers
import warnings

import numpy as np
from scipy import optimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _check_solver, _multinomial_grad_hess, _multinomial_loss, \
    _multinomial_loss_grad
from sklearn.linear_model.sag import sag_solver
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm.base import _fit_liblinear
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.optimize import newton_cg
from sklearn.utils.validation import check_X_y


class SoftLogisticRegression(LogisticRegression):
    """ LogisticRegression subclass that supports soft labels (y is 2-D matrix of probabilities) """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 verbose=0, warm_start=False):

        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter, multi_class='multinomial', verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        solver = _check_solver(self.solver, self.penalty, self.dual)

        if solver in ['newton-cg']:
            _dtype = [np.float64, np.float32]
        else:
            _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order="C", multi_output=True,
                         accept_large_sparse=solver != 'liblinear')

        self.classes_ = np.arange(y.shape[1]) if len(y.shape) > 1 else np.unique(y)

        if solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.intercept_ = np.zeros(n_classes)

        self.coef_ = logistic_regression(
            X, y, C=self.C,
            fit_intercept=self.fit_intercept, tol=self.tol,
            verbose=self.verbose, solver=solver, max_iter=self.max_iter,
            class_weight=self.class_weight, check_input=False,
            random_state=self.random_state, coef=warm_start_coef,
            penalty=self.penalty,
            max_squared_sum=max_squared_sum,
            sample_weight=sample_weight)

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self


def compute_class_weight(class_weight, y):
    """Estimate class weights for unbalanced datasets when y is soft labels

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    y : array-like, shape (n_samples, n_classes)
        Array of class probabilities per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_samples,)
        Array with class weights for each sample

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """

    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(y.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        recip_freq = len(y) / (y.shape[1] * np.sum(y))
        weight = y @ recip_freq
    else:
        raise NotImplementedError()

    return weight


def logistic_regression(
        X, y, fit_intercept=True, C=1e4, max_iter=100, tol=1e-4,
        verbose=0, solver='lbfgs', coef=None, class_weight=None, dual=False,
        penalty='l2', intercept_scaling=1., random_state=None,
        check_input=True, max_squared_sum=None, sample_weight=None):
    """Compute a Logistic Regression for possibly soft class labels y

    Based on logistic_regression_path, but assumes multinomial and removes multiple Cs logic

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    C : float
        regularization parameter that should be used. Default is 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coef : ndarray, shape (n_classes, n_features) or (n_classes, n_features + 1,)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    """

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64,
                        accept_large_sparse=solver != 'liblinear')
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    random_state = check_random_state(random_state)

    if len(y.shape) == 1:
        le = LabelBinarizer()
        y = le.fit_transform(y).astype(X.dtype, copy=False)

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    class_weight_ = compute_class_weight(class_weight, y)
    sample_weight *= class_weight_

    Y_multi = y
    nclasses = Y_multi.shape[1]

    w0 = np.zeros((nclasses, n_features + int(fit_intercept)),
                  order='F', dtype=X.dtype)

    if coef is not None:
        # it must work both giving the bias term and not

        # For binary problems coef.shape[0] should be 1, otherwise it
        # should be nclasses.
        if nclasses == 2:
            nclasses = 1

        if (coef.shape[0] != nclasses or
                coef.shape[1] not in (n_features, n_features + 1)):
            raise ValueError(
                'Initialization coef is of shape (%d, %d), expected '
                'shape (%d, %d) or (%d, %d)' % (
                    coef.shape[0], coef.shape[1], nclasses,
                    n_features, nclasses, n_features + 1))

        if nclasses == 1:
            w0[0, :coef.shape[1]] = -coef
            w0[1, :coef.shape[1]] = coef
        else:
            w0[:, :coef.shape[1]] = coef

    # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
    if solver in ['lbfgs', 'newton-cg']:
        w0 = w0.ravel()
    target = Y_multi
    warm_start_sag = {'coef': w0.T}

    if solver == 'lbfgs':
        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        iprint = [-1, 50, 1, 100, 101][
            np.searchsorted(np.array([0, 1, 2, 3]), verbose)]

        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(X, target, 1. / C, sample_weight),
            iprint=iprint, pgtol=tol, maxiter=max_iter)
        if info["warnflag"] == 1:
            warnings.warn("lbfgs failed to converge. Increase the number "
                          "of iterations.", ConvergenceWarning)

    elif solver == 'newton-cg':
        func = lambda x, *args: _multinomial_loss(x, *args)[0]
        grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
        hess = _multinomial_grad_hess

        args = (X, target, 1. / C, sample_weight)
        w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                 maxiter=max_iter, tol=tol)

    elif solver == 'liblinear':
        coef_, intercept_, n_iter_i, = _fit_liblinear(
            X, target, C, fit_intercept, intercept_scaling, None,
            penalty, dual, verbose, max_iter, tol, random_state,
            sample_weight=sample_weight)
        if fit_intercept:
            w0 = np.concatenate([coef_.ravel(), intercept_])
        else:
            w0 = coef_.ravel()

    elif solver in ['sag', 'saga']:
        target = target.astype(np.float64)
        loss = 'multinomial'
        if penalty == 'l1':
            alpha = 0.
            beta = 1. / C
        else:
            alpha = 1. / C
            beta = 0.
        w0, n_iter_i, warm_start_sag = sag_solver(
            X, target, sample_weight, loss, alpha,
            beta, max_iter, tol,
            verbose, random_state, False, max_squared_sum, warm_start_sag,
            is_saga=(solver == 'saga'))

    else:
        raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                         "'newton-cg', 'sag'}, got '%s' instead" % solver)

    nclasses = max(2, nclasses)
    multi_w0 = np.reshape(w0, (nclasses, -1))
    if nclasses == 2:
        multi_w0 = multi_w0[1][np.newaxis, :]
    coef = multi_w0.copy()

    return coef
