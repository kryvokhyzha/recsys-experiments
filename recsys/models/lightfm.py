import sklearn
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from lightfm import data

from ..dataset.column_categorizer import names_to_cats
from ..metrics import (
    test_data_preprocessing, get_result_df, users_roc_auc_scores,
    users_rr_scores, precision_at_k,
)


def get_mappings(dataset):
    """Prepare the mappings

    dataset - lightfm dataset object
    """
    return {
        'user_id_mapping': dataset.mapping()[0],
        'item_id_mapping': dataset.mapping()[2],
        'user_fename_intid_mapping': dataset._user_feature_mapping,
        'item_fename_intid_mapping': dataset._item_feature_mapping,
    }


def get_cold_weights(data_val, fename_intid_mapping):
    """Get weights for COLD users/items

    data_val - our data with features
    fename_intid_mapping - mapping with ids and category names
    """

    row_ids = []
    col_ids = []
    for ind_r, features in enumerate(data_val.values):
        for cat_name in features:
            if cat_name in fename_intid_mapping:
                row_ids.append(ind_r)
                col_ids.append(fename_intid_mapping[cat_name])
    values = [1] * len(col_ids)

    return sklearn.preprocessing.normalize(
        scipy.sparse.csr.csr_matrix(
            (values, (row_ids, col_ids)), shape=(data_val.shape[0], len(fename_intid_mapping)),
        ),
        norm="l1", copy=False
    )


def get_id_weights_mapping(
        data_val, selected_cols, id_col,
        id_mapping, fename_intid_mapping, train_weights
):
    """Get mapping with ids as keys
    and light fm sparse weights as values

    Brings hot and cold data and computes the weights matrix
    data_val - our data
    selected cols - column names WITH id column
    id_col - user/item id column
    id_mapping - light fm external-internal id mapping
    fename_intid_mapping - mapping with ids and category names
    train_weights - weights from the train time (created via dataset object)
    """

    ids_set = data_val[id_col].unique()

    id_weights_mapping = {}
    cold_ids = []
    for ind in ids_set:
        if ind in id_mapping:
            id_weights_mapping[ind] = train_weights[id_mapping[ind]]
        else:
            cold_ids.append(ind)

    if len(cold_ids) > 0:
        data_select = data_val[selected_cols]
        data_cold = data_select[data_select[id_col].isin(cold_ids)].drop_duplicates()
        cold_weights = get_cold_weights(data_cold, fename_intid_mapping)
        for ind, cold_entity in enumerate(data_cold[id_col]):
            id_weights_mapping[cold_entity] = cold_weights[ind]

    return id_weights_mapping


def get_lightfm_weights_tables_v2(
        data_val, user_id_col, item_id_col,
        user_id_weights_mapping, item_id_weights_mapping
):
    """Prepare normalized OHE sparse matrices
    Considers both HOT and COLD users/items

    data_val - our data with features
    user_id_col - users id column
    item_id_col - item id column
    user_fename_intid_mapping, item_fename_intid_mapping - mappings
    """

    result_user = []
    result_item = []
    for user_ind, item_ind in zip(data_val[user_id_col], data_val[item_id_col]):
        result_user.append(user_id_weights_mapping[user_ind])
        result_item.append(item_id_weights_mapping[item_ind])

    return scipy.sparse.vstack(result_user), scipy.sparse.vstack(result_item)


def lfm_train_preprocessing(
        data_train,
        categorizer,
        cat_cols_to_change,
        user_fe_cols,
        item_fe_cols,
        user_id_column,
        item_id_column,
        user_identity_features=True,
        item_identity_features=True
):
    """Get all the object needed for the lfm model training

    data_train - train_data
    categorizer - numeric columns categorizer object
    cat_cols_to_change - columns to change via names_to_cats() method
    user_fe_cols - user features columns WITHOUT the id columns
    item_fe_cols - item features columns WITHOUT the id columns
    user_id_column - user id column
    item_id_column - item id column
    user_identity_features - consider user id as a feature?
    item_identity_features - consider item id as a feature?
    """

    data_prep = categorizer.transform(names_to_cats(data_train, cat_cols_to_change))

    user_fe_data = data_prep[user_fe_cols].values
    item_fe_data = data_prep[item_fe_cols].values
    user_id = data_prep[user_id_column].values.ravel()
    item_id = data_prep[item_id_column].values.ravel()

    dataset = data.Dataset(
        user_identity_features=user_identity_features,
        item_identity_features=item_identity_features,
    )
    dataset.fit(
        users=user_id, items=item_id,
        user_features=user_fe_data.ravel(),
        item_features=item_fe_data.ravel(),
    )
    user_features = dataset.build_user_features(
        ((x[0], x[1]) for x in zip(user_id, user_fe_data))
    )
    item_features = dataset.build_item_features(
        ((x[0], x[1]) for x in zip(item_id, item_fe_data))
    )
    interactions, _ = dataset.build_interactions(
        ((x[0], x[1]) for x in zip(user_id, item_id))
    )

    return {
        "dataset": dataset,
        "user_features": user_features,
        "item_features": item_features,
        "interactions": interactions,
    }


def lfm_test_preprocessing(
        data_val,
        lfm_dataset,
        categorizer,
        k,
        cat_cols_to_change,
        user_fe_cols,
        item_fe_cols,
        user_id_column,
        item_id_column,
        target_col,
        train_user_weights,
        train_item_weights,
        min_prop=0.05, max_prop=0.95
):
    """Get all the objects needed for model testing

    data_val - test data
    lfm_dataset - lightfm dataset object
    categorizer - numeric columns categorizer object
    k - test_data_preprocessing() method parameter (minumum user's history size)
    cat_cols_to_change - columns to change via names_to_cats() method
    user_fe_cols - user features columns WITHOUT the id columns
    item_fe_cols - item features columns WITHOUT the id columns
    user_id_column - user id column
    item_id_column - item id column
    train_user_weights - matrix from dataset.build_user_features()
        method before the training
    train_item_weights - matrix from dataset.build_item_features()
        method before the training
    min/max_prop - minimum/maximum limits for the proportion of
        positive interactions in users histories
    """

    '''
    def test_data_preprocessing(
        data_test, user_id_col, target_col,
        min_prop=0.05, max_prop=0.95, k=10
):
    '''
    data_select = test_data_preprocessing(
        data_val, user_id_col=user_id_column,
        target_col=target_col,
        k=k, min_prop=min_prop, max_prop=max_prop
    )
    data_prep = categorizer.transform(
        names_to_cats(data_select, cat_cols_to_change)
    )
    user_cols = [user_id_column] + user_fe_cols
    item_cols = [item_id_column] + item_fe_cols

    mappings = get_mappings(lfm_dataset)
    user_id_mapping = mappings["user_id_mapping"]
    item_id_mapping = mappings["item_id_mapping"]
    user_fename_intid_mapping = mappings["user_fename_intid_mapping"]
    item_fename_intid_mapping = mappings["item_fename_intid_mapping"]

    user_id_weights_mapping = get_id_weights_mapping(
        data_prep,
        selected_cols=user_cols,
        id_col=user_id_column,
        id_mapping=user_id_mapping,
        fename_intid_mapping=user_fename_intid_mapping,
        train_weights=train_user_weights,
    )
    item_id_weights_mapping = get_id_weights_mapping(
        data_prep,
        selected_cols=item_cols,
        id_col=item_id_column,
        id_mapping=item_id_mapping,
        fename_intid_mapping=item_fename_intid_mapping,
        train_weights=train_item_weights,
    )

    u_weights, i_weights = get_lightfm_weights_tables_v2(
        data_prep,
        user_id_column,
        item_id_column,
        user_id_weights_mapping,
        item_id_weights_mapping,
    )
    ids = np.array(list(range(u_weights.shape[0])))

    return {
        "data_prep": data_prep,
        "ids": ids,
        "u_weights": u_weights,
        "i_weights": i_weights,
    }


def lfm_model_evaluation(
        model, data_val,
        categorizer,
        result_cols,
        lfm_dataset,
        train_user_weights,
        train_item_weights,
        n,
        cat_cols_to_change,
        user_fe_cols,
        item_fe_cols,
        user_id_column,
        item_id_column,
        target_col,
        score_col,
        k=5,
        num_threads=4,
        min_prop=0.05, max_prop=0.95
):
    """Light FM model evaluation method

    model - lfm model
    data_val - validation dataframe
    categorizer - numeric columns categorizer object
    result_cols - columns in the resulting dataframe (without score column)
    lfm_dataset - lightfm dataset object
    train_user_weights - matrix from dataset.build_user_features()
        method before the training
    train_item_weights - matrix from dataset.build_item_features()
        method before the training
    n - test_data_preprocessing() method parameter
    cat_cols_to_change - columns to change via names_to_cats() method
    user_fe_cols - user features columns WITHOUT the id columns
    item_fe_cols - item features columns WITHOUT the id columns
    user_id_column - user id column
    item_id_column - item id column
    target_col - target id column
    score_col - score column
    k - precision_at_k() method parameter
    num_threads - num threads for parallelization of calculations
    min/max_prop - minimum/maximum limits for the proportion of
        positive interactions in users histories
    """

    val_result_dict = lfm_test_preprocessing(
        data_val=data_val,
        lfm_dataset=lfm_dataset,
        categorizer=categorizer,
        k=n,
        cat_cols_to_change=cat_cols_to_change,
        user_fe_cols=user_fe_cols,
        item_fe_cols=item_fe_cols,
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        target_col=target_col,
        train_user_weights=train_user_weights,
        train_item_weights=train_item_weights,
        min_prop=min_prop, max_prop=max_prop,
    )

    data_val_prep = val_result_dict["data_prep"]
    val_ids = val_result_dict["ids"]
    val_u_weights = val_result_dict["u_weights"]
    val_i_weights = val_result_dict["i_weights"]

    user_items_scores_val = model.predict(
        user_ids=val_ids,
        user_features=val_u_weights,
        item_ids=val_ids,
        item_features=val_i_weights,
        num_threads=num_threads,
    )

    val_real_pred = get_result_df(data_val_prep, score_col, result_cols, user_items_scores_val)
    roc_auc_scores = users_roc_auc_scores(val_real_pred, user_id_column, target_col, score_col)
    rr_scores = users_rr_scores(val_real_pred, user_id_column, target_col, score_col)
    p_at_k_scores = precision_at_k(val_real_pred, user_id_column, target_col, score_col, k=k)

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


def get_mean_roc_auc(
        model, data_val, result_cols,
        lfm_dataset,
        categorizer,
        train_user_weights,
        train_item_weights,
        n, cat_cols_to_change,
        user_fe_cols,
        item_fe_cols,
        user_id_column,
        item_id_column,
        target_col,
        score_col,
        num_threads=4,
        min_prop=0.05, max_prop=0.95
):
    """Get mean AUC ROC for users for optimization

    model - lfm model
    data_val - validation dataframe
    result_cols - columns in the resulting dataframe (without score column)
    lfm_dataset - lightfm dataset object
    categorizer - numeric columns categorizer object
    train_user_weights - matrix from dataset.build_user_features()
        method before the training
    train_item_weights - matrix from dataset.build_item_features()
        method before the training
    n - test_data_preprocessing() method parameter
    cat_cols_to_change - columns to change via names_to_cats() method
    user_fe_cols - user features columns WITHOUT the id columns
    item_fe_cols - item features columns WITHOUT the id columns
    user_id_column - user id column
    item_id_column - item id column
    num_threads - num threads for parallelization of calculations
    min/max_prop - minimum/maximum limits for the proportion of
        positive interactions in users histories
    """

    val_result_dict = lfm_test_preprocessing(
        data_val=data_val,
        lfm_dataset=lfm_dataset,
        categorizer=categorizer,
        k=n, cat_cols_to_change=cat_cols_to_change,
        user_fe_cols=user_fe_cols,
        item_fe_cols=item_fe_cols,
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        target_col=target_col,
        train_user_weights=train_user_weights,
        train_item_weights=train_item_weights,
        min_prop=min_prop, max_prop=max_prop
    )

    data_val_prep = val_result_dict["data_prep"]
    val_ids = val_result_dict["ids"]
    val_u_weights = val_result_dict["u_weights"]
    val_i_weights = val_result_dict["i_weights"]

    user_items_scores_val = model.predict(
        user_ids=val_ids,
        user_features=val_u_weights,
        item_ids=val_ids,
        item_features=val_i_weights,
        num_threads=num_threads,
    )

    val_real_pred = get_result_df(data_val_prep, score_col, result_cols, user_items_scores_val)
    roc_auc_scores = users_roc_auc_scores(val_real_pred, user_id_column, target_col, score_col)

    return np.mean(roc_auc_scores)


def get_embeddings_s(lfm_model, item_id_weights_mapping):
    """Get scaled items embeddings from a sparse weights mapping
    and the lightfm model

    lfm_model - light fm model
    item_id_weights_mapping - weights created via light fm dataset() object
    """

    result = []
    for key in item_id_weights_mapping:
        item_sparse_weights = item_id_weights_mapping[key]
        item_vec = lfm_model.get_item_representations(item_sparse_weights)[1][0]
        result.append(item_vec)
    result_s = StandardScaler().fit_transform(np.array(result))
    return result_s


def vis_lfm_embeddings(
        data_in, lfm_model, categorizer, reducer, selected_cols,
        item_id_column, cat_cols_to_change, smpl_size,
        mappings, train_weights, popular_cls_num,
        figsize, alpha, cluster_col
):
    """Plot embeddings with reduced dimension

    data_in - initial dataframe
    lfm_model - light fm model
    categorizer - numeric columns categorizer object
    reducer - dimension reducer
    selected_cols - item features columns + id column
    item_id_column - item id column
    cat_cols_to_change - columns to change via names_to_cats() method
    smpl_size - the number of objects to plot embeddings for
    mappings - mappings from the light fm dataset object
    train_weights - matrix from dataset.build_item_features()
        method before the training
    popular_cls_num - number of classes to differ
    figsize - final picture size
    alpha - transparency coef
    cluster_col - the column to use for clustering, so
        different values -> different colors
    """

    warnings.filterwarnings("ignore")
    unique_data = data_in[selected_cols].drop_duplicates()[:smpl_size].copy()
    data_prep = categorizer.transform(
        names_to_cats(unique_data, cat_cols_to_change)
    )
    item_id_weights_mapping = get_id_weights_mapping(
        data_prep,
        selected_cols=selected_cols,
        id_col=item_id_column,
        id_mapping=mappings["item_id_mapping"],
        fename_intid_mapping=mappings["item_fename_intid_mapping"],
        train_weights=train_weights,
    )

    embeddings = get_embeddings_s(lfm_model, item_id_weights_mapping)
    X = reducer.fit_transform(embeddings)

    most_popular_clusters = list(data_prep[cluster_col].value_counts()[:popular_cls_num].index)
    reduced_df = pd.DataFrame({
        'c1': X[:, 0],
        'c2': X[:, 1],
        cluster_col: data_prep[cluster_col].values
    })
    reduced_df["cls"] = reduced_df[cluster_col].apply(lambda x: x if x in most_popular_clusters else "other")

    plt.figure(figsize=figsize)
    sns.scatterplot(data=reduced_df, x="c1", y="c2", hue="cls", alpha=alpha)
    plt.grid();
