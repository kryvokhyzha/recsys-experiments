from sklearn.base import TransformerMixin, BaseEstimator


class RandomRanker(TransformerMixin, BaseEstimator):
    """Ranker that maps zeros as scores to emulate random algorithm
    !!! we need shuffled data !!!
    """

    def get_score(self, data):
        """Map zeros as scores to emulate random algorithm

        data - test dataset
        """

        data_new = data.copy()
        data_new["score"] = [0] * data_new.shape[0]
        return data_new
