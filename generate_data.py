import math
import random

import keras

from initial_conditions import fid, fprop, fsin
import numpy as np

from dump import save
from f_estimators import naive, split_sequence
from gmm import GMM
from initial_conditions import num_model, series_per_model, ts_len, gmm_components, train_test_split
from model import Model
from nn import Architecture
from process import ProbabilisticCharacterization
import matplotlib.pyplot as plt

save_it = True

if __name__ == '__main__':
    models = []
    X = list()
    Y = list()
    k = 0
    fs = [fid, fprop, fsin]
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
    models = np.array(models)
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

    X_train, y_train = WX[:math.floor((len(WX) * (1 - train_test_split)))], Y[:math.floor(
        (len(WX) * (1 - train_test_split)))]
    X_test, y_test = WX[math.floor(len(WX) * (1 - train_test_split)):], Y[
                                                                        math.floor(len(WX) * (1 - train_test_split)):]
    gmm_train, gmm_test = gmm_features[:math.floor((len(WX) * (1 - train_test_split)))], gmm_features[math.floor(
        len(WX) * (1 - train_test_split)):]
    print(X_train.shape, gmm_train.shape, y_train.shape)
    print(X_test.shape, gmm_test.shape, y_test.shape)

    if save_it:
        save(X, Y, WX, models, X_train, y_train, X_test, y_test, gmm_train, gmm_test)

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
    model.save('dumps/model.keras')
