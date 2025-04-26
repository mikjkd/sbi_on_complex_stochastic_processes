import math
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from f_estimators import naive, split_sequence
from gmm import GMM
from model import Model
from nn import Architecture
from process import ProbabilisticCharacterization

# from f_estimators import *


num_model = 3
series_per_model = 50
ts_len = 300
gmm_components = 2
hidden_lstm_space = 10

if __name__ == "__main__":
    models = []
    X = list()
    Y = list()
    k = 0
    fs = [lambda x: x, lambda x: 1.3 * x, lambda x: math.sin(x)]
    for i in range(num_model):
        models.append(Model(
            theta=[ProbabilisticCharacterization(mu=random.randint(-10, 10), sigma=random.uniform(0, 2)),
                   ProbabilisticCharacterization(mu=random.randint(-10, 10), sigma=random.uniform(0, 2))], f=fs[i]),
        )
        print(
            f'model{i}: mus {models[i].theta[0].mu},{models[i].theta[1].mu}, sigmas {models[i].theta[0].sigma},{models[i].theta[1].sigma}')
        for j in range(series_per_model):
            X.append(models[i].generate(ts_len))
            Y.append(i)

    indices = list(range(len(X)))
    random.shuffle(indices)
    X = np.array(X)
    Y = np.array(Y)
    X = X[indices]
    Y = Y[indices]
    # xi = np.diff(X, axis=1)
    xi = list()
    for x in X:
        xi.append(naive(x))
    xi = np.array(np.real(xi))
    # qui conserviamo le features per ogni ts
    # NON conosciamo la ts a quale modello appartiene
    # ma sappiamo che all'i-imo x in X appartiene l'imo feature vector in  gmm_features
    gmm = GMM(k=gmm_components)
    gmm_features = []
    for i in range(len(xi)):
        gmm.train(xi[i].reshape(-1, 1))
        mus = gmm.model.means_[:, 0]
        sigmas = np.sqrt(gmm.model.covariances_[:, 0, 0])
        print(f'model {Y[i]}, mus {mus}, sigmas {sigmas}')
        gmm_features.append(np.array([mus, sigmas]))
    gmm_features = np.array(gmm_features)
    # Ogni x in X deve essere finestrata
    WX = []
    WY = []
    for idx, x in enumerate(X):
        wx, _ = split_sequence(x, n_steps=30, n_steps_y=0)
        y = np.array([Y[idx]] * len(wx))
        WX.append(wx)
    WX = np.array(WX)
    print(X.shape, Y.shape, WX.shape, gmm_features.shape)
    # (10, 1000) (10,) (10, 971, 30) (10, 971) (10, 2, 2)
    train_test_split = 0.2

    X_train, y_train = WX[:math.floor((len(WX) * (1 - train_test_split)))], Y[:math.floor(
        (len(WX) * (1 - train_test_split)))]
    X_test, y_test = WX[math.floor(len(WX) * (1 - train_test_split)):], Y[
                                                                        math.floor(len(WX) * (1 - train_test_split)):]
    gmm_train, gmm_test = gmm_features[:math.floor((len(WX) * (1 - train_test_split)))], gmm_features[math.floor(
        len(WX) * (1 - train_test_split)):]
    print(X_train.shape, gmm_train.shape, y_train.shape)
    print(X_test.shape, gmm_test.shape, y_test.shape)

    model = Architecture(lstm_input_shape=(X_train.shape[1], X_train.shape[2]),
                         gmm_input_shape=(gmm_train.shape[1], gmm_train.shape[2]),
                         embedding_space=30,
                         num_model=num_model
                         )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
    history = model.fit([X_train, gmm_train], y_train, epochs=100, batch_size=32, validation_split=0.2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('tests/accuracy_plot.pdf')  # salva come immagine
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('tests/loss_plot.pdf')
    plt.close()

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
