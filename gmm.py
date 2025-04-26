from sklearn.mixture import GaussianMixture


class GMM:
    def __init__(self, k):
        self.n_comp = k
        self.model = GaussianMixture(n_components=k, covariance_type='full', random_state=42)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)