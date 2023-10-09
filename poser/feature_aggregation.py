"""Functions necessary for aggregating temporal features into single values."""
import logging
import time
from collections import defaultdict
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import preprocessing
import nolds
from pycatch22 import catch22_all


from .utils import NestedPool

L = logging.getLogger(__name__)


def aggregate(
    patient_collection,
    behavioural_state=None,
    n_workers=1,
    use_catch22=False,
    normalize_features=False,
    use_dummies=False,
    compute_advanced=False,
):
    """Main function to extract features over time.

    Args:
        patient_collection (PatientCollection object): PatientCollection object with loaded patients (see patients.py)
        behavioural_state (int): an integer defining the behavioural state (this requires the time-series to have been labelled into behavioural states)
        n_workers (int): number of workers for parallel processing
        use_catch22 (bool): Whether to compute catch22 features (see catch22 package)
        normalize_features (bool): Whether to normalize features or not
        use_dummies (bool): Whether to use the dummy variables (e.g. sum of hidden markers)

    Returns:
        (DataFrame): dataframe of features
    """

    L.info(
        "Computing aggregate features for %s patients (we disabled %s patients).",
        len(patient_collection),
        patient_collection.get_num_disabled_patients(),
    )

    features_df = compute_all_features(
        patient_collection,
        behavioural_state=behavioural_state,
        n_workers=n_workers,
        use_catch22=use_catch22,
        normalize_features=normalize_features,
        use_dummies=use_dummies,
        compute_advanced=compute_advanced,
    )

    L.info("%s features extracted.", features_df.shape[1])

    if normalize_features:
        features_df = normalize(features_df)

    features_df = features_df.sort_index()

    return features_df


def normalize(features_df):
    """ Normalize the aggregated features """
    col_names = features_df.columns
    scaler = preprocessing.StandardScaler()
    features_df = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(features_df, columns=col_names)
    return features_df     


def aggregate_features(patient, 
                       behavioural_state=None, 
                       normalize_features=False, 
                       use_catch22=False, 
                       use_dummies=True, 
                       compute_advanced=False,
                       sub_name='',
                       ):
    """Extract features for a single patient

    Args:
        patients (GraphCollection object): GraphCollection object with loaded patients (see patients.py)
        feature_dict (Dictionary): Dictionary of features to compute

    Returns:
        (DataFrame): dataframe (single row) of calculated features for a given patient.
    """

    features_df = pd.DataFrame()
    structural_features = patient.structural_features.drop('time',axis=1)        
    struct_feats = structural_features.columns
    markers = patient.markers
    
    if behavioural_state is not None:
        structural_features = structural_features.loc[(patient.movement_labels_df.label==behavioural_state),:]
    
    if structural_features.empty:
        structural_features.loc[0,structural_features.columns] = 0

    # calculate mean
    feat_name = '_mean' + sub_name
    features_df.loc[patient.patient_id,struct_feats+feat_name] = np.nanmean(structural_features,axis=0)
    
    # calculate median
    feat_name = '_median' + sub_name
    features_df.loc[patient.patient_id,struct_feats+feat_name] = np.nanmedian(structural_features,axis=0)
    
    # calculate std
    feat_name = '_std' + sub_name
    features_df.loc[patient.patient_id,struct_feats+feat_name] = np.nanstd(structural_features,axis=0)

    # calculate max
    feat_name = '_max' + sub_name
    features_df.loc[patient.patient_id,struct_feats+feat_name] = np.nanmax(structural_features,axis=0) 

    if compute_advanced:

        # calculate min
        feat_name = '_min' + sub_name
        features_df.loc[patient.patient_id,struct_feats+feat_name] = np.nanmin(structural_features,axis=0) 
        # calculate entropy
        feat_name = '_entropy' + sub_name
        features_df.loc[patient.patient_id,struct_feats+feat_name] = approximate_entropy(structural_features)


    # calculate catch22
    if use_catch22:
        features_df = pd.concat([features_df, compute_catch22(structural_features, patient.patient_id)], axis=1)

    # get # na values
    if use_dummies:
        feat_name = '_sum_visible' + sub_name
        features_df.loc[patient.patient_id,pd.Index(markers)+feat_name] = (patient.dummy_variables==0).sum(axis=0).values

    if patient.label is not None:
        if isinstance(patient.label, dict):
            for key in patient.label:
                features_df.loc[patient.patient_id,key] = patient.label[key]
        else:
            features_df.loc[patient.patient_id,'label'] = patient.label

    return features_df

def compute_all_features(
    patients,
    behavioural_state=None,
    n_workers=1,
    use_catch22=False,
    normalize_features=False,
    use_dummies=False,
    compute_advanced=False,
):
    """Compute features for all patients

    Args:
        patients (GraphCollection object): GraphCollection object with loaded patients (see patients.py)
        feature_dict (Dictionary): Dictionary of features to compute
        n_workers (int): number of workers for parallel processing

    Returns:
        (List): list of patients with computed feature dataframes.
    """

    L.info("Computing features for %s patients:", len(patients))
    
    with NestedPool(n_workers) as pool:
        return pd.concat(
            tqdm(
                pool.imap(
                    partial(
                        aggregate_features,
                        behavioural_state=behavioural_state,
                        normalize_features=normalize_features,
                        use_catch22=use_catch22,
                        use_dummies=use_dummies,
                        compute_advanced=compute_advanced,
                    ),
                    patients,
                ),
                total=len(patients),
            )
        )



def compute_catch22(structural_features, patient_id):
    """ Computing catch22 features """
    catch22_features_df = pd.DataFrame()

    for feat in structural_features.columns:
        x = structural_features.loc[:,feat].dropna()
        catch22_computation = catch22_all(x)
        
        features = catch22_computation['values']    
        feature_names = catch22_computation['names']
        feature_names = [feat + '_' + catch22_feat for catch22_feat in feature_names]
        
        catch22_features_df.loc[patient_id,feature_names] = features
    return catch22_features_df

def approximate_entropy(structural_features, remove_nan=True):
    """ Computing approximate entropy of timeseries """    
    
    entropies = np.zeros([structural_features.shape[1]])
    for i,feature in enumerate(structural_features.columns):
        timeseries = structural_features[feature]

        if remove_nan:
            timeseries = timeseries.dropna()
        
        if timeseries.shape[0]>3:
            entropy = nolds.sampen(timeseries)
        else: 
            entropy = np.nan
            
        entropies[i] = entropy
    
    return entropies
