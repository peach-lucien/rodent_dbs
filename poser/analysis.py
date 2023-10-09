"""
Functions for the analysis of extracted feature matrix.
"""
import itertools
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_curve
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import pearsonr

from tqdm import tqdm

from .pio import load_fitted_model, save_fitted_model
from .plotting import plot_analysis, plot_prediction

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
warnings.simplefilter("ignore")


def features_to_Xy(features):
    """Decompose features dataframe to numpy arrays X and y."""
    if "label" in features:
        return features.drop(columns=["label"]), features["label"]

    return features, None


def _normalise_feature_data(features, scaler=None, fit_scaler=True):
    """Normalise the feature matrix to remove the mean and scale to unit variance."""
    if "label" in features:
        label = features["label"]

    if scaler is None:
        scaler = StandardScaler()
        normed_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    else:
        if fit_scaler:
            normed_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        else:
            normed_features = pd.DataFrame(scaler.transform(features), columns=features.columns)

    normed_features.index = features.index

    if "label" in features:
        normed_features["label"] = label

    return normed_features, scaler


def _number_folds(y):
    """Get number of folds."""
    counts = y.value_counts()
    L.info("Counts of patients/label: \n%s", counts)
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


def _get_reduced_feature_set(X, shap_top_features, n_top_features=100, alpha=0.9):
    """Reduce the feature set by taking uncorrelated features."""
    rank_feat_ids = np.argsort(shap_top_features)[::-1]

    selected_features = [rank_feat_ids[0]]
    corr_matrix = np.abs(np.corrcoef(X.T))
    for rank_feat_id in rank_feat_ids[1:]:
        if not (corr_matrix[selected_features, rank_feat_id] > alpha).any():
            selected_features.append(rank_feat_id)
        if len(selected_features) >= n_top_features:
            break

    L.info(
        "Now using a reduced set of %s features with < %s correlation.",
        str(len(selected_features)),
        str(alpha),
    )
    if len(selected_features) == 1:
        L.warning("Only one features selected, things may break!")

    return X.columns[selected_features]


def _print_accuracy(acc_scores, analysis_type, reduced=False):
    """Print the classification or regression accuracies."""
    if reduced:
        if analysis_type == "classification":
            L.info(
                "Accuracy with reduced set: %s +/- %s",
                str(np.round(np.mean(acc_scores), 3)),
                str(np.round(np.std(acc_scores), 3)),
            )
        elif analysis_type == "regression":
            L.info(
                "Mean Absolute Error with reduced set: %s +/- %s",
                str(np.round(np.mean(acc_scores), 3)),
                str(np.round(np.std(acc_scores), 3)),
            )

    else:
        if analysis_type == "classification":
            L.info(
                "Accuracy: %s +/- %s",
                str(np.round(np.mean(acc_scores), 3)),
                str(np.round(np.std(acc_scores), 3)),
            )
        elif analysis_type == "regression":
            L.info(
                "Mean absolute error: %s +/- %s",
                str(np.round(np.mean(acc_scores), 3)),
                str(np.round(np.std(acc_scores), 3)),
            )


def _filter_patients(features, patient_removal=0.05):
    """Remove samples with more than X% bad values."""
    n_patients = len(features.index)
    features = features[features.isnull().sum(axis=1) / len(features.columns) < patient_removal]
    L.info(
        "%s patients were removed for more than %s fraction of bad features",
        str(n_patients - len(features.index)),
        str(patient_removal),
    )
    
    if 'label' in features:
        n_patients_nolabel = features['label'].isna().sum()
        features = features[features.label.notna()]
        L.info(
            "%s/%s patients were removed for missing label",
            str(n_patients_nolabel),
            str(n_patients),
        )
        
    return features


def _filter_features(features):
    """Filter features and create feature matrix."""
    nan_features = features.replace([np.inf, -np.inf], np.nan)
    valid_features = nan_features.dropna(axis=1).astype("float64")
    return valid_features.drop(
        valid_features.std()[(valid_features.std() == 0)].index, axis=1
    ).columns


def _get_model(model, analysis_type):
    """Get a model."""
    if isinstance(model, str):
        if model == "RF":
            if analysis_type == "classification":
                from sklearn.ensemble import RandomForestClassifier

                L.info("... Using RandomForest classifier ...")
                model = RandomForestClassifier(n_estimators=1000, max_depth=30)
            if analysis_type == "regression":
                from sklearn.ensemble import RandomForestRegressor

                L.info("... Using RandomForest regressor ...")
                model = RandomForestRegressor(n_estimators=1000, max_depth=30)

        elif model == "XG":
            if analysis_type == "classification":
                from xgboost import XGBClassifier

                L.info("... Using Xgboost classifier ...")
                model = XGBClassifier(eval_metric="mlogloss")
            if analysis_type == "regression":
                from xgboost import XGBRegressor

                L.info("... Using Xgboost regressor ...")
                model = XGBRegressor(objective="reg:squarederror", eval_metric="mlogloss")
        elif model == "NN":
            if analysis_type == "classification":
                from sklearn.neural_network import MLPClassifier

                L.info("... Using neural network classifier ...")
                model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 4), random_state=42)
            if analysis_type == "regression":
                from sklearn.neural_network import MLPRegressor

                L.info("... Using neural network regressor ...")
                model = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 4), random_state=42)
        elif model == "linear":
            if analysis_type == "classification":
                from sklearn.linear_model import LogisticRegression
                L.info("... Using Logistic regression ...")
                model= LogisticRegression()
            if analysis_type == "regression":
                from sklearn.linear_model import LinearRegression
                L.info("... Using Linear regression ...")
                model= LinearRegression()
        else:
            raise Exception("Unknown model type: {}".format(model))
    return model


def _get_shap_feature_importance(shap_values):
    """From a list of shap values per folds, compute the global shap feature importance."""
    # average across folds
    mean_shap_values = np.mean(shap_values, axis=0)
    # average across labels
    if len(np.shape(mean_shap_values)) > 2:
        global_mean_shap_values = np.mean(mean_shap_values, axis=0)
        mean_shap_values = list(mean_shap_values)
    else:
        global_mean_shap_values = mean_shap_values

    # average accros patients
    shap_feature_importance = np.mean(abs(global_mean_shap_values), axis=0)
    return mean_shap_values, shap_feature_importance


def _evaluate_kfold(X, y, model, folds, analysis_type, compute_shap, bootstrap):
    """Evaluate the kfolds."""
    shap_values = []
    acc_scores = []
    roc_curves = []
    for indices in folds.split(X, y=y):
        acc_score, shap_value, roc_curve_ = _compute_fold(X, y, model, 
                                                          indices, 
                                                          analysis_type,
                                                          compute_shap,
                                                          bootstrap,)
        shap_values.append(shap_value)
        acc_scores.append(acc_score)
        roc_curves.append(roc_curve_)
        
    return acc_scores, shap_values, roc_curves


def fit_model_kfold(
    features,
    model,
    analysis_type="classification",
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    n_repeats=1,
    random_state=42,
    n_splits=None,
    compute_shap=True,
    bootstrap=False,
):
    """Classify patients from extracted features with kfold.

    Args:
        features (dataframe): extracted features
        model (str): model to preform analysis
        analysis_type (str): 'classification' or 'regression'
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        n_repeats (int): number of k-fold repeats
        random_state (int): rng seed
        n_splits (int): numbere of split for k-fold, None=automatic estimation
        compute_shap (bool): compute SHAP values or not

    Returns:
        (dict): dictionary with results
    """
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = features_to_Xy(features)
    
    if analysis_type == "classification":
        if n_splits is None:
            n_splits = _number_folds(y)
        L.info("Using %s splits", str(n_splits))
        folds = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
     
        
    elif analysis_type == "regression":
        if n_splits is None:
            n_splits = _number_folds(y)
        L.info("Using %s splits", str(n_splits))
        folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    acc_scores, shap_values, roc_curves = _evaluate_kfold(X, y, model, folds, analysis_type, compute_shap, bootstrap)
    _print_accuracy(acc_scores, analysis_type)

    if compute_shap:
        mean_shap_values, shap_feature_importance = _get_shap_feature_importance(shap_values)
    else:
        mean_shap_values = None
        shap_feature_importance = None

    analysis_results = {
        "X": X,
        "y": y,
        "acc_scores": acc_scores,
        "mean_shap_values": mean_shap_values,
        "shap_values": shap_values,
        "shap_feature_importance": shap_feature_importance,
        "reduced_features": None,
        "roc_curves": roc_curves, 
    }
    if not reduce_set:
        return analysis_results

    if not compute_shap:
        return analysis_results

    reduced_features = _get_reduced_feature_set(
        X,
        shap_feature_importance,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )
    reduced_acc_scores, reduced_shap_values, roc_curves = _evaluate_kfold(
        X[reduced_features], y, model, folds, analysis_type, compute_shap, bootstrap,
    )
    _print_accuracy(reduced_acc_scores, analysis_type, reduced=True)
    (
        reduced_mean_shap_values,
        reduced_shap_feature_importance,
    ) = _get_shap_feature_importance(reduced_shap_values)

    analysis_results.update(
        {
            "reduced_features": reduced_features,
            "reduced_shap_values": reduced_shap_values,
            "shap_values": shap_values,
            "reduced_acc_scores": reduced_acc_scores,
            "reduced_mean_shap_values": reduced_mean_shap_values,
            "reduced_shap_feature_importance": reduced_shap_feature_importance,
            "roc_curves": roc_curves, 
        }
    )
    return analysis_results


def fit_model(
    features,
    model,
    analysis_type="classification",
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    test_size=0.2,
    random_state=42,
    compute_shap=True,
    bootstrap=False,
):
    """Train a single model.

    Args:
        features (dataframe): extracted features
        model (str): model to preform analysis
        analysis_type (str): 'classification' or 'regression'
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        random_state (int): rng seed
        test_size (float): size of test dataset (see sklearn.model_selection.ShuffleSplit)
        compute_shap (bool): compute SHAP values or not

    Returns:
        (dict): dictionary with results
    """
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = features_to_Xy(features)

    indices = next(ShuffleSplit(test_size=test_size, random_state=random_state).split(X, y))
    acc_score, shap_values, _ = _compute_fold(X, y, model, indices, analysis_type, compute_shap, bootstrap)

    mean_shap_values, shap_feature_importance = _get_shap_feature_importance([shap_values])
    analysis_results = {
        "X": X,
        "y": y,
        "acc_score": acc_score,
        "mean_shap_values": mean_shap_values,
        "shap_feature_importance": shap_feature_importance,
        "model": model,
        "indices": indices,
        "reduced_features": None,
    }
    if not reduce_set:
        return analysis_results

    reduced_features = _get_reduced_feature_set(
        X,
        shap_feature_importance,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )
    reduced_model = deepcopy(model)
    reduced_acc_score, reduced_shap_values, roc_curve_ = _compute_fold(
        X[reduced_features], y, reduced_model, indices, analysis_type, compute_shap, bootstrap
    )
    (
        reduced_mean_shap_values,
        reduced_shap_feature_importance,
    ) = _get_shap_feature_importance(reduced_shap_values)

    analysis_results.update(
        {
            "reduced_features": reduced_features,
            "reduced_acc_score": reduced_acc_score,
            "reduced_mean_shap_values": reduced_mean_shap_values,
            "reduced_shap_feature_importance": reduced_shap_feature_importance,
            "reduced_model": reduced_model,
            "roc_curves": roc_curve_
        }
    )
    return analysis_results

def _bootstrap_data(y, train_index, random_state=42):
    """ upsample smaller classes """
    unique_classes, count = np.unique(y[train_index],return_counts=True)
    n_upsample = count.max()
    
    train_index_bootstraps = []
    y_bootstraps = []
    for i in range(unique_classes.shape[0]):        
        class_train_index = train_index[y[train_index]==unique_classes[i]]        
        if count[i] < n_upsample:
            
            resampling = resample(class_train_index,
                                 n_samples=n_upsample,
                                 random_state=random_state
                                 )
            train_index_bootstraps.append(resampling
                                          )
            y_bootstraps.append(y[resampling])
        else:
            train_index_bootstraps.append(class_train_index)
            y_bootstraps.append(y[class_train_index])
            
    return pd.Series(np.hstack(y_bootstraps)), np.hstack(train_index_bootstraps)

def _compute_fold(X, y, model, indices, analysis_type, compute_shap, bootstrap):
    """Compute a single fold for parallel computation."""
    
    train_index, val_index = indices

    if bootstrap:
        y, train_index = _bootstrap_data(y, train_index)
    
    model.fit(X.iloc[train_index], y.iloc[train_index])

    if analysis_type == "classification":
        acc_score = accuracy_score(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Fold accuracy: --- %s ---", str(np.round(acc_score, 3)))
        
    elif analysis_type == "regression":
        acc_score = mean_absolute_error(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Mean Absolute Error: --- %s ---", str(np.round(acc_score, 3)))
        
        r,p = pearsonr(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Pearson r: --- %s ---", str(np.round(r, 3)))

    try:
        _roc_curve = roc_curve(y.iloc[val_index], model.predict(X.iloc[val_index]))#, 2)
    except:
        _roc_curve = None

    if compute_shap:
        if (model.__class__ in [x[1] for x in inspect.getmembers(sklearn.linear_model,inspect.isclass)]):
            explainer = shap.LinearExplainer(model,X.iloc[train_index])#,feature_perturbation='correlation_dependent')
        else:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:  # pylint: disable=broad-except
                explainer = shap.KernelExplainer(model.predict_proba, X.iloc[train_index])
        shap_values = explainer.shap_values(X)
    else:
        shap_values = None

    return acc_score, shap_values, _roc_curve


def _preprocess_features(
    features, patient_removal, trained_model=None
):
    """Collect all feature filters."""
    L.info("%s total features", str(len(features.columns)))

    features = _filter_patients(features, patient_removal=patient_removal)

    good_features = _filter_features(features)
    features = features[good_features]

    scaler = None
    if trained_model is not None:
        scaler = trained_model[1]
        # missing_cols = model_features.columns[~model_features.columns.isin(features.columns)]
        # for _col in missing_cols:
        #     features[_col] = np.nan
        # features = features[model_features.columns]

    L.info("%s valid features", str(len(features.columns)))

    features, scaler = _normalise_feature_data(features, scaler=scaler)

    return features, scaler


def train_all(features, model):
    """Train on all available data."""
    X, y = features_to_Xy(features)
    model.fit(X, y)
    L.info("Fitting model to all data")
    return model


def predict_unlabelled(trained_model, features):
    """Predict unlabelled data."""
    model = trained_model[0]
    L.info("Using %s features from pre-trained model", str(len(trained_model[2])))
    features = features.loc[:,trained_model[2]]
    X, _ = features_to_Xy(features)
    return model.predict(X), model.predict_proba(X)


def analysis(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    features,
    patients=None,
    analysis_type="classification",
    folder="../../examples/outputs/",
    patient_removal=0.3,
    model="XG",
    compute_shap=True,
    kfold=True,
    reduce_set=True,
    reduced_set_size=4,
    reduced_set_max_correlation=0.9,
    plot=True,
    max_feats_plot=20,
    max_feats_plot_dendrogram=100,
    n_repeats=1,
    n_splits=None,
    random_state=42,
    test_size=0.2,
    trained_model=None,
    save_model=False,
    bootstrap=False,
):
    """Main function to classify patients and plot results.

    Args:
        features (dataframe): extracted features
        patients (PatientCollection): input patients
        analysis_type (str): 'classification' or 'regression'
        folder (str): folder to save analysis
        patient_removal (float): remove samples with more than patient_removal % bad values
        model (str): model to preform analysis
        compute_shap (bool): compute SHAP values or not
        kfold (bool): run with kfold
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        plot (bool): save plots
        max_feats_plot (int): max number of feature analysis to plot
        n_repeats (int): number of k-fold repeats
        n_splits (int): numbere of split for k-fold, None=automatic estimation
        random_state (int): rng seed
        test_size (float): size of test dataset (see sklearn.model_selection.ShuffleSplit)
        trained_model (str): provide path to pretrained model to apply to new data
        save_modeel (bool): save the obtained model to reuse later

    Returns:
        (dict): dictionary with results
    """

    if trained_model is None:
        model = _get_model(model, analysis_type)
    else:
        if isinstance(trained_model, tuple):
            model = trained_model[0]
        elif isinstance(trained_model, str):
            trained_model = load_fitted_model(trained_model)
            model = trained_model[0]

    features, scaler = _preprocess_features(
        features, patient_removal, trained_model
    )

    if not Path(folder).exists():
        os.mkdir(folder)

    if trained_model is not None:
        y_predictions, y_proba = predict_unlabelled(trained_model, features)
        _save_predictions_to_csv(features, y_predictions, y_proba, folder=folder)
        
        if 'label' in features:
            acc_score = accuracy_score(y_predictions, features.label)
            L.info("Prediction accuracy: --- %s ---", str(np.round(acc_score, 3)))    
            
        return y_predictions, y_proba

    if kfold:
        analysis_results = fit_model_kfold(
            features,
            model,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
            compute_shap=compute_shap,
            bootstrap=bootstrap,
        )
    else:
        analysis_results = fit_model(
            features,
            model,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            random_state=random_state,
            test_size=test_size,
            compute_shap=compute_shap,
            bootstrap=bootstrap,
        )

    if save_model:
         fitted_model = train_all(features, model)

    if analysis_type == "regression":
        analysis_results["mean_shap_values"] = [analysis_results["mean_shap_values"]]
        if "reduced_mean_shap_values" in analysis_results:
            analysis_results["reduced_mean_shap_values"] = [
                analysis_results["reduced_mean_shap_values"]
            ]

    results_folder = Path(folder) / ("results_plots_")
    if not Path(results_folder).exists():
        os.mkdir(results_folder)

    if compute_shap:
        if plot and kfold:
            plot_analysis(
                analysis_results,
                results_folder,
                patients,
                analysis_type,
                max_feats_plot,
                max_feats_plot_dendrogram,
            )
        if plot and not kfold:
            plot_prediction(analysis_results, results_folder)

        if kfold:
            _save_to_csv(analysis_results, features, results_folder)



    if save_model:
        model_folder = Path(folder) / ("fitted_model")
        if not Path(model_folder).exists():
            os.mkdir(model_folder)
        save_fitted_model(fitted_model, scaler, list(features.columns.drop('label')),  model_folder)

    return analysis_results


def _save_predictions_to_csv(features, predictions, probability, folder="results"):
    """Save the prediction results for unlabelled data."""
    col_names = ['probability_class_{}'.format(c) for c in range(probability.shape[1])]
    results_df = pd.DataFrame(data=np.vstack([predictions,probability.T]).T, 
                              index=features.index,
                              columns=["y_prediction"] + col_names)
    results_df.to_csv(os.path.join(folder, "prediction_results.csv"))


def _save_to_csv(analysis_results, features, folder="results"):
    """Save csv file with analysis data."""

    result_df = pd.DataFrame(columns=analysis_results["X"].columns)
    result_df.loc["shap_feature_importance"] = analysis_results["shap_feature_importance"]
    shap_values = analysis_results["shap_values"]
    for i, shap_class in enumerate(shap_values):
        result_df.loc["shap_importance: class {}".format(i)] = np.vstack(shap_class).mean(axis=0)

    if analysis_results["reduced_features"] is not None:
        reduced_features = analysis_results["reduced_features"]
        result_df.loc["reduced_shap_feature_importance", reduced_features] = analysis_results[
            "reduced_shap_feature_importance"
        ]
        shap_values = analysis_results["reduced_shap_values"]
        for i, shap_class in enumerate(shap_values):
            result_df.loc[
                "reduced shap_importance: class {}".format(i), reduced_features
            ] = np.vstack(shap_class).mean(axis=0)

        result_df = result_df.sort_values(
            "reduced_shap_feature_importance", axis=1, ascending=False
        )
    else:
        result_df = result_df.sort_values("shap_feature_importance", axis=1, ascending=False)

    result_df.to_csv(os.path.join(folder, "importance_results.csv"))


def classify_pairwise(  # pylint: disable=too-many-locals
    features,
    model="XG",
    patient_removal=0.3,
    n_top_features=5,
    reduce_set=False,
    reduced_set_size=100,
    reduced_set_max_correlation=0.5,
    n_repeats=1,
    n_splits=None,
    analysis_type="classification",
):
    """Classify all possible pairs of clases with kfold and returns top features.

    The top features for each pair with high enough accuracies are collected in a list,
    for later analysis.

    Args:
        features (dataframe): extracted features
        model (str): model to preform analysis
        patient_removal (float): remove samples with more than patient_removal % bad values
        n_top_features (int): number of top features to save
        reduce_set (bool): is True, the classification will be rerun
            on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
            in reduced set of features
        n_repeats (int): number of k-fold repeats
        n_splits (int): numbere of split for k-fold, None=automatic estimation
        analysis_type (str): 'classification' or 'regression'

    Returns:
        (dataframe, list, int): accuracies dataframe, list of top features, number of top pairs
    """
    features, _ = _preprocess_features(
        features, patient_removal, 
    )

    classifier = _get_model(model, analysis_type=analysis_type)
    classes = features.label.unique()
    class_pairs = list(itertools.combinations(classes, 2))
    accuracy_matrix = pd.DataFrame(columns=classes, index=classes)

    top_features = {}
    for pair in tqdm(class_pairs):
        features_pair = features.loc[(features.label == pair[0]) | (features.label == pair[1])]
        analysis_results = fit_model_kfold(
            features_pair,
            classifier,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            n_splits=n_splits,
        )
        if "reduced_acc_scores" in analysis_results:
            accuracy_matrix.loc[pair[0], pair[1]] = np.round(
                np.mean(analysis_results["reduced_acc_scores"]), 3
            )
        else:
            accuracy_matrix.loc[pair[0], pair[1]] = np.round(
                np.mean(analysis_results["acc_scores"]), 3
            )
        accuracy_matrix.loc[pair[1], pair[0]] = accuracy_matrix.loc[pair[0], pair[1]]

        if "reduced_shap_feature_importance" in analysis_results:
            top_feat_idx = analysis_results["reduced_shap_feature_importance"].argsort()[
                -n_top_features:
            ]
            top_features_raw = analysis_results["X"][analysis_results["reduced_features"]].columns[
                top_feat_idx
            ]
        else:
            top_feat_idx = analysis_results["shap_feature_importance"].argsort()[-n_top_features:]
            top_features_raw = analysis_results["X"].columns[top_feat_idx]

        top_features[pair] = [f_class + "_" + f for f_class, f in top_features_raw]

    return accuracy_matrix.astype(float), top_features