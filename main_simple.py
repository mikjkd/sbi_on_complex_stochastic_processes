import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from dump import load
from initial_conditions import train_test_split

# from f_estimators import *

if __name__ == "__main__":

    X, Y, WX, models, X_train, y_train, X_test, y_test, gmm_train, gmm_test, model = load()

    model.evaluate([X_test, gmm_test], y_test)
    y_hats = model.predict([X_test, gmm_test])
    y_hats = y_hats.argmax(axis=1)

    for idx, x in enumerate(X_test):
        y_hat = y_hats[idx]
        plt.plot(X[math.floor((len(WX) * (1 - train_test_split))) + idx], label=f'pred: {y_hat}, true: {y_test[idx]}')
        plt.legend()
        plt.savefig(f'tests/test_{idx}.pdf')
        plt.close()

        # show difference between learned mu/sigma vs true mu/sigma
        true_model = y_test[idx]
        predicted_model = y_hat
        fig, axs = plt.subplots(2, figsize=(20, 6))
        fig.suptitle(f'True pdf vs learned pdf for model {true_model}')
        mu1_true = models[true_model].theta[0].mu
        mu2_true = models[true_model].theta[1].mu
        sigma1_true = models[true_model].theta[0].sigma
        sigma2_true = models[true_model].theta[1].sigma
        x1_true = np.linspace(mu1_true - 3 * sigma1_true, mu1_true + 3 * sigma1_true, 100)
        x2_true = np.linspace(mu2_true - 3 * sigma2_true, mu2_true + 3 * sigma2_true, 100)
        axs[0].plot(x1_true, stats.norm.pdf(x1_true, mu1_true, sigma1_true))
        axs[0].plot(x2_true, stats.norm.pdf(x2_true, mu2_true, sigma2_true))

        mu1_pred = gmm_test[idx][0][0]
        mu2_pred = gmm_test[idx][0][1]
        sigma1_pred = gmm_test[idx][1][0]
        sigma2_pred = gmm_test[idx][1][1]
        x1_pred = np.linspace(mu1_pred - 3 * sigma1_pred, mu1_pred + 3 * sigma1_pred, 100)
        x2_pred = np.linspace(mu2_pred - 3 * sigma2_pred, mu2_pred + 3 * sigma2_pred, 100)
        axs[1].plot(x1_pred, stats.norm.pdf(x1_pred, mu1_pred, sigma1_pred))
        axs[1].plot(x2_pred, stats.norm.pdf(x2_pred, mu2_pred, sigma2_pred))
        fig.savefig(f'tests/pdf_{idx}.pdf')
        plt.close(fig)
