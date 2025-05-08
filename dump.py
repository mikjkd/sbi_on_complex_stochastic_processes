import pickle as pkl

import keras


def save(X, Y, WX, models, X_train, y_train, X_test, y_test, gmm_train, gmm_test):
    with open("dumps/train.pkl", "wb") as f:
        pkl.dump([X_train, y_train], f)
    with open("dumps/test.pkl", "wb") as f:
        pkl.dump([X_test, y_test], f)
    with open("dumps/gmm_train.pkl", "wb") as f:
        pkl.dump(gmm_train, f)
    with open("dumps/gmm_test.pkl", "wb") as f:
        pkl.dump(gmm_test, f)
    with open("dumps/models.pkl", "wb") as f:
        pkl.dump(models, f)
    with open("dumps/XY.pkl", "wb") as f:
        pkl.dump([X, Y], f)
    with open("dumps/WX.pkl", "wb") as f:
        pkl.dump(WX, f)


def load():
    with open("dumps/train.pkl", "rb") as f:
        X_train, y_train = pkl.load(f)
    with open("dumps/test.pkl", "rb") as f:
        X_test, y_test = pkl.load(f)
    with open("dumps/gmm_train.pkl", "rb") as f:
        gmm_train = pkl.load(f)
    with open("dumps/gmm_test.pkl", "rb") as f:
        gmm_test = pkl.load(f)
    with open("dumps/models.pkl", "rb") as f:
        models = pkl.load(f)
    with open("dumps/WX.pkl", "rb") as f:
        WX = pkl.load(f)
    with open("dumps/XY.pkl", "rb") as f:
        X, Y = pkl.load(f)

    model = keras.saving.load_model('dumps/model.keras')
    return X, Y, WX, models, X_train, y_train, X_test, y_test, gmm_train, gmm_test, model
