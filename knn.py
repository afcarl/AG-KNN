from sklearn.neighbors import BallTree
import pandas as pd
from collections import Counter
import numpy as np


class KNN():
    def __init__(self, k, weighted=False):
        """ Constructs object. """
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        """ Fits the model.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe of coordinates to query
        y : pd.DataFrame
            Dataframe of metadata

        Returns
        -------
        self
        """
        assert X.shape[0] == y.shape[0]
        self.obs = X.reindex(index=y.index)
        self.tree = BallTree(X.values)
        self.metadata = y
        return self

    def query(self, X):
        """ Queries neighbors

        Parameters
        ----------
        X : pd.DataFrame
           Dataframe of coordinates to query

        Returns
        -------
        dist : np.array, float
           Distances to each neighbor
        md : pd.DataFrame
           Dataframe of metadata for each neighbor
        """

        dist, ind = self.tree.query(X.values, k=self.k)
        for i in range(X.shape[0]):
            md = self.metadata.iloc[ind[i].ravel()]
            yield dist[i], md

    def predict(self, X):
        """ Makes prediction

        Parameters
        ----------
        X : pd.DataFrame
           Dataframe of coordinates to query

        Returns
        -------
        pred : pd.DataFrame
           Dataframe of predicted metadata for each
           query.
        """
        pred = pd.DataFrame(
            columns=self.metadata.columns,
            index=X.index)
        gen = self.query(X)
        # Need to figure out how to aggregate findings.
        for i, (dist, md) in enumerate(gen):
            x = X.index[i]
            for c in self.metadata.columns:
                col = md[c].copy()
                col = col.dropna().values
                try:
                    # recognize that it is a floating point
                    col = col.astype(np.float)
                    p = md[c].median()
                    if np.isnan(p):
                        p = 0
                    pred.loc[x, c] = p
                except:
                    votes = pd.Series(Counter(md[c]))
                    pred.loc[x, c] = np.argmax(votes)
        return pred


def compare(obs_md, pred_md):
    res = pd.Series(columns=['type', 'measure'],
        index=self.metadata.columns)
    for c in obs_md.columns:
        obs_col = obs_md[c].copy()
        pred_col = pred_md[c].copy()
        try:
            # recognize that it is a floating point
            col = col.astype(np.float)
            res.loc[c, 'measure'] = np.mean(np.abs(obs_col - pred_col))
            res.loc[c, 'type'] = 'continuous'
        except:
            res.loc[c, 'measure'] = (obs_col == pred_col).astype(np.int) / len(obs_col)
            res.loc[c, 'type'] = 'categorical'
    return res


