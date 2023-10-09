"""Functions necessary for the extraction of features still functionalised by time."""
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

from .utils import NestedPool

from .structural_features import compute_distances, compute_angles, compute_areas, compute_ratios, compute_marker_dynamics

L = logging.getLogger(__name__)



def extract(
    patient_collection,
    structural_features,
    n_workers=1,
    normalize_features=False,
):
    """Main function to extract features over time.

    Args:
        patient_collection (PatientCollection object): PatientCollection object with loaded patients (see patients.py)
        structural_features (StructuralFeatures object): StructuralFeatures object with dictionary of features to compute.
        n_workers (int): number of workers for parallel processing
        normalize_features (bool): standard normalise features

    Returns:
        patient_collection (PatientCollection object): PatientCollection object with loaded patients (see patients.py). Each patient object inside the 
                                                        PatientCollection now has a DataFrame with structural features.
    """


    L.info(
        "Extracting features from %s patients (we disabled %s patients).",
        len(patient_collection),
        patient_collection.get_num_disabled_patients(),
    )

    patients_ = compute_all_features(
        patient_collection,
        structural_features.features,
        n_workers=n_workers,
        normalize=normalize_features,
    )

    patient_collection.patients = patients_
    L.info("%s feature extracted.", len(patients_[0].structural_features.columns))
    return patient_collection


def normalize(structural_features_df):
    col_names = structural_features_df.columns
    scaler = preprocessing.StandardScaler()
    structural_features_df = scaler.fit_transform(structural_features_df)
    structural_features_df = pd.DataFrame(structural_features_df, columns=col_names)
    return structural_features_df
    

def feature_extraction(patient, feature_dict, pose_index=None, normalize_features=False):
    """Extract features for a single patient

    Args:
        patients (GraphCollection object): GraphCollection object with loaded patients (see patients.py)
        feature_dict (Dictionary): Dictionary of features to compute

    Returns:
        (DataFrame): dataframe of calculated features for a given patient.
    """
    angles_df = pd.DataFrame()
    areas_df = pd.DataFrame()
    markers_df = pd.DataFrame()
    
    if pose_index is None:
        pose_estimation = patient.pose_estimation
    else:
        pose_estimation = patient.pose_estimation.loc[pose_index[0]:pose_index[1],:]

    # if empty then compute all pairwise distances
    if not bool([True for a in feature_dict.values() if a]):
        distances_df = compute_distances(pose_estimation, patient.markers)
    else:
        distances_df = compute_distances(pose_estimation, feature_dict['distances'])

    if feature_dict['angles']:
        angles_df = compute_angles(pose_estimation, feature_dict['angles'])
    if feature_dict['areas']:
        areas_df = compute_areas(pose_estimation, feature_dict['areas'])
    if feature_dict['markers']:
        markers_df = compute_marker_dynamics(pose_estimation,
                                             feature_dict['markers'],
                                             patient.sampling_frequency,
                                             )
        
    structural_features_df = pd.concat([distances_df, angles_df, areas_df, markers_df],axis=1)

    if feature_dict['ratios']:
        ratios_df = compute_ratios(structural_features_df, feature_dict['ratios'])
        structural_features_df = pd.concat([structural_features_df, ratios_df],axis=1)


    if normalize_features:
        structural_features_df = normalize(structural_features_df)

    structural_features_df['time'] = pose_estimation['time'].values
    patient.structural_features = structural_features_df

    return patient



def compute_all_features(
    patients,
    feature_dict,
    n_workers=1,
    normalize=True,
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
        return list(
            tqdm(
                pool.imap(
                    partial(
                        feature_extraction,
                        feature_dict=feature_dict,
                        normalize_features=normalize,
                    ),
                    patients,
                ),
                total=len(patients),
            )
        )


