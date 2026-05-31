import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


def generate_clean_data(n_samples=500, noise=20, random_state=42):
    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, true_coef


def add_outliers(X, y, n_outliers=25, random_state=42):
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    y_out = y.copy()
    X_out[:n_outliers] = 10 + 0.75 * rng.randn(n_outliers, X.shape[1])
    y_out[:n_outliers] = -15 + 20 * rng.randn(n_outliers)
    return X_out, y_out


def plot_dataset_with_outliers(X, y, n_outliers=25):
    fig, ax = plt.subplots()
    ax.scatter(X[n_outliers:], y[n_outliers:], color='steelblue', s=10, label='Normal observations')
    ax.scatter(X[:n_outliers], y[:n_outliers], color='tomato', s=30, marker='x', label='Artificial outliers')
    ax.set_title('Dataset with Outliers')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig


def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_huber_regression(X, y):
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    return {name: abs(coef - true_coef) for name, coef in coef_dict.items()}


def best_robust_model(errors):
    robust = {k: v for k, v in errors.items() if k != 'linear_regression'}
    return min(robust, key=robust.get)


def ransac_outlier_summary(X, y, n_outliers=25, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask
    total_outliers_detected = int(outlier_mask.sum())
    added_outliers_detected = int(outlier_mask[:n_outliers].sum())
    return total_outliers_detected, added_outliers_detected


def plot_regression_fits(X, y, random_state=42):
    lr = fit_linear_regression(X, y)
    huber = fit_huber_regression(X, y)
    ransac = fit_ransac_regression(X, y, random_state)
    theilsen = fit_theilsen_regression(X, y, random_state)

    X_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    def predict_line(coef):
        model = LinearRegression()
        model.fit(X, y)
        # Use direct coef for lines
        return X_line.ravel() * coef

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='lightgray', s=10, label='Data')

    x_vals = X_line.ravel()
    for coef, label, color in [
        (lr, 'Linear Regression', 'red'),
        (huber, 'Huber', 'gold'),
        (ransac, 'RANSAC', 'green'),
        (theilsen, 'Theil-Sen', 'blue'),
    ]:
        ax.plot(x_vals, x_vals * coef, color=color, linewidth=2, label=label)

    ax.set_title('Regression Model Fits')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig


def plot_ransac_inliers_outliers(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    inlier_mask = model.inlier_mask_

    fig, ax = plt.subplots()
    ax.scatter(X[inlier_mask], y[inlier_mask], color='steelblue', s=10, label='Inliers')
    ax.scatter(X[~inlier_mask], y[~inlier_mask], color='tomato', s=30, marker='x', label='Outliers')
    ax.set_title('RANSAC: Inliers vs Outliers')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig