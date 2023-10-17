import numpy as np
import numpy.ma as ma


def nan_to_num(X, y, nan_mode):
    if nan_mode is None:
        indices = ~np.isnan(X).any(axis=(1, 2))
        X, y = X[indices], y[indices]
    elif nan_mode == 0:
        X = np.nan_to_num(X, nan=0.0)
    elif nan_mode == "avg":
        mask = ma.array(X, mask=np.isnan(X)).mean(axis=(0, 2))[:, np.newaxis]
        X = np.where(np.isnan(X), mask, X)
    return X, y


class Normalizer:
    scale = "scale"
    z_score = "z_score"
    
    def __init__(self, X_min=0.0, X_max=0.0, X_std=0.0, X_avg=0.0, vals=None):
        if vals is not None:
            self.X_min, self.X_max, self.X_std, self.X_avg = vals
        else:
            self.X_min = X_min
            self.X_max = X_max
            self.X_std = X_std
            self.X_avg = X_avg
    
    def fit(self, X):
        self.X_min = np.min(X, axis=(0, 2))[:, np.newaxis]
        self.X_max = np.max(X, axis=(0, 2))[:, np.newaxis]
        self.X_std = np.std(X, axis=(0, 2))[:, np.newaxis]
        self.X_avg = np.average(X, axis=(0, 2))[:, np.newaxis]
        return self.X_min, self.X_max, self.X_std, self.X_avg
    
    def fit_transform(self, X, mode):
        self.fit(X)
        X_norm = self.transform(X, mode)
        return X_norm
    
    def transform(self, X, mode):
        if mode == Normalizer.scale:
            return (X - self.X_min) / (self.X_max - self.X_min)
        else:
            return (X - self.X_avg) / self.X_std
    
    def values(self):
        return np.array([self.X_min, self.X_max, self.X_std, self.X_avg])
    
    def __repr__(self):
        return (f"{Normalizer.__name__}(min: {self.X_min}, max:{self.X_max},"
                f"avg: {self.X_avg}, std: {self.X_std})")
