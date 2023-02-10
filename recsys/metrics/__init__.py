import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def get_result_df(data_val, score_col, result_cols, scores):
    """Get resulting dataframe

    data_val - initial data with features
    score_col - score column name
    result_cols - columns without scores (ids, other useful info)
    scores - predicted scores
    """

    real_pred_df = data_val[result_cols].copy()
    real_pred_df[score_col] = scores
    return real_pred_df


def users_roc_auc_scores(
        real_pred_data, user_id_col,
        target_col, score_col,
):
    """Compute ROC AUC scores
    for users

    real_pred_data - data with target and score
    user_id_col - user_id column
    target_col, score_col - columns with target values and model's scores
    """

    roc_auc_scores = []
    for _, data_us in real_pred_data.groupby(user_id_col):
        roc_auc_scores.append(roc_auc_score(data_us[target_col], data_us[score_col]))
    return roc_auc_scores


def users_rr_scores(
        real_pred_data, user_id_col,
        target_col, score_col,
):
    """Compute RR scores for users

    real_pred_data - data with target and score
    user_id_col - user_id column
    target_col, score_col - columns with target values and model's scores
    """

    rr_scores = []
    for _, data_us in real_pred_data.groupby(user_id_col):
        data_us_sorted = data_us.sort_values(by=[score_col], ascending=False) \
            .reset_index(drop=True)
        rr_score = 1 / ((data_us_sorted[target_col] == 1).argmax() + 1)
        rr_scores.append(rr_score)
    return rr_scores


def precision_at_k(
        real_pred_data, user_id_col,
        target_col, score_col, k=5,
):
    """Compute precision@k scores for users

    real_pred_data - data with target and score
    k - size of the top
    user_id_col - user_id column
    target_col, score_col - columns with target values and model's scores
    """

    precision_scores = []
    for _, data_us in real_pred_data.groupby(user_id_col):
        data_us_sorted = data_us.sort_values(by=[score_col], ascending=False).iloc[:k]
        precision_score = data_us_sorted[target_col].sum() / k

        precision_scores.append(precision_score)
    return precision_scores


def test_data_preprocessing(
        data_test, user_id_col, target_col,
        min_prop=0.05, max_prop=0.95, k=10
):
    """Pick only users
    with both positive and negative responses

    data_test - data to transform
    user_id_col - id column name
    k - minimum number of songs that a user interacted with
    min/max_prop - minimum/maximum limits for the proportion of
        positive interactions in users histories
    """

    data_new = data_test.copy()
    val_users = []
    for user_id, data_us in data_new.groupby([user_id_col]):
        pos_cnt = data_us[target_col].sum()
        cnt = data_us.shape[0]
        pos_prop = pos_cnt / cnt
        if cnt >= k and min_prop < pos_prop < max_prop:
            val_users.append(user_id)

    data_prep = data_new[data_new[user_id_col].isin(val_users)].reset_index(drop=True)
    return data_prep


def model_evaluation(
        model, data_val, result_cols, user_id_col, target_col, score_col, n,
        min_prop=0.05, max_prop=0.95, k=5,
):
    """Baseline models evaluation function

    model - PopularRanker model
    data_val - validation dataframe
    result_cols - columns form initial dataset to consider in final
    user_id_col - user id column name
    n - minimum number of songs that a user interacted with
    k - size of the top
    min/max_prop - minimum/maximum limits for the proportion of
        positive interactions in users histories
    """

    data_val_prep = test_data_preprocessing(
        data_val[result_cols], user_id_col=user_id_col, target_col=target_col,
        k=n, min_prop=min_prop, max_prop=max_prop,
    )
    val_real_pred = model.get_score(data_val_prep)
    roc_auc_scores = users_roc_auc_scores(
        val_real_pred, user_id_col=user_id_col,
        target_col=target_col, score_col=score_col,
    )
    rr_scores = users_rr_scores(
        val_real_pred, user_id_col=user_id_col,
        target_col=target_col, score_col=score_col,
    )
    p_at_k_scores = precision_at_k(
        val_real_pred, user_id_col=user_id_col,
        target_col=target_col, score_col=score_col, k=k,
    )
    print(f"Users mean AUC ROC on test: {np.mean(roc_auc_scores)}")
    print(f"Users mean reciprocal rank on test: {np.mean(rr_scores)}")
    print(f"Users mean precision@{k} on test: {np.mean(p_at_k_scores)}")
    plt.figure()
    plt.hist(roc_auc_scores)
    plt.title("USERS AUC ROC SCORES")
    plt.grid()
    plt.figure()
    plt.hist(rr_scores)
    plt.title(f"USERS RECIPROCAL RANK SCORES")
    plt.grid()
    plt.figure()
    plt.hist(p_at_k_scores)
    plt.title(f"USERS PRECISION@{k} SCORES")
    plt.grid()
    return val_real_pred


def plot_histories_stats(
        data_in, k, user_id_col,
        item_id_col,
        result_cols,  # =result_cols+["genre_ids"]
        target_col,
        min_prop=0.25, max_prop=0.75,
        prepare=True
):
    """Explore data stats

    data_in - data to explore before preprocessing
    k - user history min size
    user_id_col/item_id_col - user/item id column
    result_cols - columns form initial dataset to consider in final
    min_prop/max_prop - min/max positive proportions to consider
    prepare - TRUE/FALSE, use test_data_preprocessing?
    """
    data_prep = data_in.copy()

    if prepare:
        data_prep = test_data_preprocessing(
            data_in[result_cols], user_id_col=user_id_col, target_col=target_col,
            k=k, min_prop=min_prop, max_prop=max_prop
        )

    users_history_size_list = []
    users_history_prop_list = []
    users_history_unique_prop_list = []
    for _, data_us in data_prep.groupby(user_id_col):
        pos_cnt = data_us[target_col].sum()
        unique_items_count = data_us[item_id_col].unique().shape[0]
        cnt = data_us.shape[0]
        pos_prop = pos_cnt / cnt
        unique_item_prop = unique_items_count / cnt
        users_history_size_list.append(cnt)
        users_history_prop_list.append(pos_prop)
        users_history_unique_prop_list.append(unique_item_prop)
    plt.figure()
    plt.hist(users_history_size_list)
    plt.title("USERS HISTORIES SIZES")
    plt.grid()
    plt.figure()
    plt.hist(users_history_prop_list)
    plt.title(f"USERS HISTORIES POS PROPORTIONS")
    plt.grid()
    plt.figure()
    plt.hist(users_history_unique_prop_list)
    plt.title(f"USERS HISTORIES UNIQUE ITEMS PROPORTIONS")
    plt.grid()
    plt.figure()
    data_prep["genre_ids"].value_counts()[:10].plot(kind="bar")
    plt.title(f"10 POPULAR GENRES DISTRIBUTION")
    plt.grid()
