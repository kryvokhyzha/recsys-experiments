from sklearn.base import TransformerMixin, BaseEstimator


class PopularRanker(TransformerMixin, BaseEstimator):
    """Simple ranker based on value counts of
    users positive responces
    """

    def __init__(self, column_to_agg):
        self.column_to_agg = column_to_agg

    def fit(self, data):
        """Compute counts

        data - training dataset with only POSITIVE responces
        """

        self.counts = data[self.column_to_agg].value_counts().to_dict()
        return self

    def get_score(self, data):
        """Map the counts to the given items ids

        data - test dataset
        """

        data_new = data.copy()
        data_new["score"] = data_new[self.column_to_agg].map(self.counts).fillna(-1).values
        return data_new
