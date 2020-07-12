import cytoolz as tz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score

from moe import mixture, logistic


def example():
    n = 1000
    d = 2
    k = 3
    std_noise = .2

    num_experts = 3

    from sklearn.linear_model import LinearRegression
    expert = LinearRegression()

    # from sklearn.tree import DecisionTreeRegressor
    # expert = DecisionTreeRegressor(max_leaf_nodes=5)

    X, y, true_model = generate_data(n, k, d, std_noise)

    model, yhat = fit_model(X, y, expert, num_experts)
    print('R^2:', r2_score(y, yhat))
    visualize_fit(X, model, true_model, y, yhat)

    return model


def visualize_fit(X, model, true_model, y, yhat):
    plot_predictions(X, y, true_model, title='True')
    plot_predictions(X, y, model, title='Pred')
    plot_predictions(X, y, model.gate, title='Gate')


def fit_model(X, y, expert, num_experts):
    experts = [clone(expert) for _ in range(num_experts)]
    gate = logistic.SoftLogisticRegression()
    model = mixture.MixtureOfExpertsRegressor(experts, gate, tol=1e-8)
    model.fit(X, y)
    yhat = model.predict(X)
    return model, yhat


def generate_data(n, k, d, std_noise):
    X = np.random.normal(size=(n, d))
    betas = np.random.normal(size=(d, k))
    gate_betas = np.random.normal(size=(d, k))
    noise = np.random.normal(size=n) * std_noise
    true_model = tz.partial(predict_noiseless, betas=betas, gate_betas=gate_betas)
    y = true_model(X) + noise
    return X, y, true_model


def predict_noiseless(X, betas, gate_betas):
    yall = X @ betas
    exp_weight = np.exp(X @ gate_betas)
    mask = exp_weight == np.max(exp_weight, axis=1)[:, None]
    mask = mask / mask.sum(axis=1)[:, None]
    y = (yall * mask).sum(axis=1)
    return y


def plot_predictions(X, y, model=None, **kwargs):
    """ plot 2D data with colored background representing function over space """
    plt.figure()

    if model is not None:
        h = .025
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Xgrid = np.c_[xx.ravel(), yy.ravel()]

        # obtain scores / labels for each point in mesh
        if callable(model):
            Z = model(Xgrid)
        else:
            Z = model.predict(Xgrid)

        Z = Z.reshape(xx.shape)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   # cmap=plt.cm.brg,
                   aspect='auto', origin='lower')

    plt.scatter(*X.T, c=y)
    ax = plt.gca()
    ax.set(**kwargs)
    plt.show(block=False)
    return ax
