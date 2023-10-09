"""plotting functions."""
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import auc



matplotlib.use("Agg")
L = logging.getLogger(__name__)


def _save_to_pdf(pdf, figs=None):
    """Save a list of figures to a pdf."""
    if figs is not None:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")
    else:
        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot_prediction(analysis_results, folder, ext=".png", figsize=(6, 6)):
    """Plot the prediction of the trained model."""

    def _plot_roc_curves(analysis_results):
        #plt.figure()
        mean_fpr = np.linspace(0, 1, 100)
        tprs=[]
        aucs =[] 
        for i in range(len(analysis_results['roc_curves'])):
            fpr = analysis_results['roc_curves'][i][0]
            tpr = analysis_results['roc_curves'][i][1]
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(
                    fpr,
                    tpr, 
                    lw=1, 
                    alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
                    )
            
        plt.plot(
            [0, 1],
                 [0, 1],
                 linestyle='--',
                 lw=2, color='r',
                 label='Chance',
                 alpha=.8
                 )
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate',fontsize=8)
        plt.ylabel('True Positive Rate',fontsize=8)
        plt.title('Cross-Validation ROC',fontsize=8)
        plt.legend(loc="lower right", prop={'size': 8})

    def _plot_pred(analysis_results, model, X, acc):
        train_index, test_index = analysis_results["indices"]
        prediction_test = model.predict(X.iloc[test_index])
        prediction_train = model.predict(X.iloc[train_index])
        #plt.figure(figsize=figsize)
        plt.plot(
            analysis_results["y"].iloc[test_index],
            prediction_test,
            "+",
            c="C0",
            label="test samples",
        )
        plt.plot(
            analysis_results["y"].iloc[train_index],
            prediction_train,
            ".",
            c="0.5",
            label="train samples",
        )
        plt.plot(analysis_results["y"], analysis_results["y"], ls="--", c="k")
        plt.legend(loc="best")
        plt.suptitle("Accuracy: " + str(np.round(acc, 2)))
        plt.xlabel("Value")
        plt.ylabel("Predicted value")

    figs = []
    if "model" in analysis_results:
        figs.append(plt.figure(figsize=figsize))
        _plot_pred(
            analysis_results,
            analysis_results["model"],
            analysis_results["X"],
            analysis_results["acc_score"],
        )
        plt.savefig(os.path.join(folder, "prediction" + ext), dpi=200, bbox_inches="tight")

    if "roc_curves" in analysis_results:
        if analysis_results['roc_curves'][0] is not None:
            figs.append(plt.figure(figsize=figsize))
            _plot_roc_curves(
                analysis_results,
            )
            plt.savefig(
                os.path.join(folder, "roc_curve" + ext),
                dpi=200,
                bbox_inches="tight",
            )

    if "reduced_model" in analysis_results:
        figs.append(plt.figure(figsize=figsize))
        _plot_pred(
            analysis_results,
            analysis_results["reduced_model"],
            analysis_results["X"][analysis_results["reduced_features"]],
            analysis_results["reduced_acc_score"],
        )
        plt.savefig(
            os.path.join(folder, "reduced_prediction" + ext),
            dpi=200,
            bbox_inches="tight",
        )
    
    return figs


def plot_analysis(
    analysis_results,
    folder,
    patients,
    analysis_type,
    max_feats=20,
    max_feats_dendrogram=100,
    ext=".svg",
):
    """Plot summary of poser analysis."""
    if "reduced_mean_shap_values" in analysis_results:
        L.info("Using reduced features for plotting.")
        shap_values = analysis_results["reduced_mean_shap_values"]
        reduced_features = analysis_results["reduced_features"]
        feature_importance = analysis_results["reduced_shap_feature_importance"]
    else:
        shap_values = analysis_results["mean_shap_values"]
        reduced_features = analysis_results["X"].columns
        feature_importance = analysis_results["shap_feature_importance"]

    X = analysis_results["X"][reduced_features]
    y = analysis_results["y"]

    with PdfPages(os.path.join(folder, "analysis_report.pdf")) as pdf:
        
        L.info("Plot prediction")
        figs = plot_prediction(analysis_results, folder, ext=ext)
        _save_to_pdf(pdf, figs)
       
        L.info("Plot bar ranking")
        _bar_ranking_plot(shap_values, X, folder, max_feats, ext=ext)
        _save_to_pdf(pdf)

        L.info("Plot dot summary")
        figs = _dot_summary_plot(shap_values, X, folder, max_feats, ext=ext)
        _save_to_pdf(pdf, figs)

        L.info("Plot feature correlations")
        _plot_feature_correlation(
            analysis_results["shap_feature_importance"],
            analysis_results["X"],
            reduced_features,
            folder,
            max_feats_dendrogram,
            ext=ext,
        )
        _save_to_pdf(pdf)

        L.info("Plot dendrogram")
        _plot_dendrogram_shap(
            analysis_results["shap_feature_importance"],
            analysis_results["X"],
            reduced_features,
            folder,
            max_feats_dendrogram,
            ext=ext,
        )
        _save_to_pdf(pdf)

        if analysis_type == "classification":
            L.info("Plot shap violin")
            _plot_shap_violin(feature_importance, X, y, folder, max_feats, ext=ext)
        elif analysis_type == "regression":
            L.info("Plot trend")
            _plot_trend(feature_importance, X, y, folder, max_feats, ext=ext)
        _save_to_pdf(pdf)

        if patients is not None:
            L.info("Plot feature summaries")
            # figs = _plot_feature_summary(
            #     X[reduced_features], y, patients, folder, shap_values, max_feats, ext=ext
            # )
            #_save_to_pdf(pdf, figs)


def _bar_ranking_plot(mean_shap_values, X, folder, max_feats, ext=".png"):
    """Function for customizing and saving SHAP summary bar plot."""
    
    shap.summary_plot(mean_shap_values, X, plot_type="bar", max_display=max_feats, show=False)
    
    plt.title("Feature Rankings-All Classes")
    plt.savefig(os.path.join(folder, "shap_bar_rank" + ext), dpi=200, bbox_inches="tight")


def _dot_summary_plot(shap_values, data, folder, max_feats, ext=".png"):
    """Function for customizing and saving SHAP summary dot plot."""
    num_classes = len(shap_values)
    if len(np.shape(shap_values)) == 2:
        shap_values = [shap_values]
        num_classes = 1

    figs = []
    for i in range(num_classes):
        figs.append(plt.figure())
        shap.summary_plot(shap_values[i], data, plot_type="dot", max_display=max_feats, show=False)
        plt.title("Sample Expanded Feature Summary for Class " + str(i))
        plt.savefig(
            os.path.join(folder, "shap_class_{}_summary{}".format(i, ext)),
            dpi=200,
            bbox_inches="tight",
        )
    return figs


def _plot_dendrogram_shap(
    shap_feature_importance, X, reduced_features, folder, max_feats, ext=".png"
):
    """Plot dendrogram witth hierarchical clustering."""
    top_feat_idx = shap_feature_importance.argsort()[-max_feats:]
    X_red = X[X.columns[top_feat_idx]]
    # to make sure to have the reduced features
    X_red = X_red.T.append(X[reduced_features].T).drop_duplicates().T

    plt.figure(figsize=(20, 1.2 * 20))
    gs = GridSpec(2, 1, height_ratios=[0.2, 1.0])
    gs.update(hspace=0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    cor = np.abs(X_red.corr())
    Z = linkage(cor.to_numpy(), "ward")
    dn = dendrogram(Z, labels=X_red.columns, ax=ax1)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel("Euclidean Distance")

    top_feats_names = X_red.columns[dn["leaves"]]
    cor_sorted = np.abs(X_red[top_feats_names].corr())

    cb_axis = inset_axes(ax1, width="5%", height="12%", borderpad=2, loc="upper right")
    sns.heatmap(
        cor_sorted,
        square=True,
        ax=ax2,
        linewidth=0.5,
        cbar_ax=cb_axis,
        cbar_kws={"label": "|pearson|"},
    )
    cor_sorted["id"] = np.arange(len(cor_sorted))
    reduced_pos = cor_sorted.loc[cor_sorted.index.intersection(reduced_features), ["id"]] + 0.5
    ax2.scatter(reduced_pos, reduced_pos, c="g", s=100)
    ax1.title.set_text("Top {} features heatmap and dendogram".format(max_feats))
    plt.savefig(os.path.join(folder, "shap_dendogram" + ext), dpi=200, bbox_inches="tight")


def _plot_feature_correlation(
    shap_feature_importance, X, reduced_features, folder, max_feats, ext=".png"
):
    """Plot correlation matrix."""
    top_feat_idx = shap_feature_importance.argsort()[-max_feats:]
    X_red = X[X.columns[top_feat_idx]].sort_index(axis=0).sort_index(axis=1)
    # to make sure to have the reduced features
    X_red = X_red.T.append(X[reduced_features].T).drop_duplicates().T
    cor_sorted = np.abs(X_red.corr())

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    cb_axis = inset_axes(ax, width="5%", height="20%", borderpad=2, loc="upper right")
    sns.heatmap(
        cor_sorted,
        square=True,
        ax=ax,
        linewidth=0.5,
        cbar_ax=cb_axis,
        cbar_kws={"label": "|pearson|"},
    )
    cor_sorted["id"] = np.arange(len(cor_sorted))
    #print(cor_sorted.loc[cor_sorted.index.intersection(reduced_features), ["id"]])
    reduced_pos = cor_sorted.loc[cor_sorted.index.intersection(reduced_features), ["id"]] + 0.5
    ax.scatter(reduced_pos, reduced_pos, c="g", s=100)
    plt.savefig(os.path.join(folder, "correlation_matrix" + ext), dpi=200, bbox_inches="tight")


PERCENTILES = [2, 25, 50, 75, 98]



def _plot_shap_violin(shap_feature_importance, data, labels, folder, max_feats, ext=".png"):
    """Plot the violins of a feature."""
    top_feat_idx = shap_feature_importance.argsort()[-max_feats:]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values

        data_split = []
        for k in np.unique(labels):
            indices = np.argwhere(labels.values == k)
            data_split.append(feature_data[indices])

        sns.violinplot(data=data_split, ax=ax, palette="muted", width=1)
        ax.set(xlabel="Class label", ylabel=data.columns[top_feat])

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_violins" + ext),
        dpi=200,
        bbox_inches="tight",
    )


def _plot_trend(shap_feature_importance, data, labels, folder, max_feats, ext=".png"):
    """Plot the violins of a feature."""
    top_feat_idx = shap_feature_importance.argsort()[-max_feats:]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values

        sns.scatterplot(x=feature_data, y=labels, ax=ax, palette="muted")
        ax.set(xlabel=data.columns[top_feat], ylabel="y-label")

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_trend" + ext),
        dpi=200,
        bbox_inches="tight",
    )


def pca_plot(features, pca):
    """Plot pca of data."""
    X = pca.transform(features)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
def plot_gif(patient, folder='./outputs/gifs/'):
    """ create gif of movement """
    
    temp_folder = './temp/'
    os.mkdir(temp_folder)

    x_ = patient.pose_estimation.loc[:,pd.IndexSlice[:, 'x']]
    y_ = -patient.pose_estimation.loc[:,pd.IndexSlice[:, 'y']]    
    
    xe_ = patient.embedded_structural_features[0]
    ye_ = patient.embedded_structural_features[1]
    ze_ = patient.embedded_structural_features[2]
        
    filenames = []
    skip_number = 2
    
    for i in range(0,patient.pose_estimation.shape[0],skip_number):
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(2, 2, 1)    
        for marker in patient.markers:
            ax1.scatter(x_.loc[i,marker],
                        y_.loc[i,marker],
                        label=marker)
            ax1.set_xlim([x_.min().min(), x_.max().max()])
            ax1.set_ylim([y_.min().min(), y_.max().max()])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')     
            
           
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')   
        ax2.scatter(xe_[:i+1],ye_[:i+1],ze_[:i+1], c = patient.embedded_structural_features.loc[:i,'cluster_id'])
        ax2.set_xlim([xe_.min(),xe_.max()])
        ax2.set_ylim([ye_.min(),ye_.max()])
        ax2.set_zlim([ze_.min(),ze_.max()])
        ax2.view_init(elev=10., azim=i*360/(patient.pose_estimation.shape[0]/skip_number))
        #ax2.text(1, 1, str(embedded_features.loc[i,'cluster_id']), size=15)
        ax2.set_xlabel('TSNE-1')
        ax2.set_ylabel('TSNE-2')
        ax2.set_zlabel('TSNE-3')
        
        
        ax3 = fig.add_subplot(2, 2, 3)   
        ax3.bar(range(7),patient.cluster_proba_[i,:])#,colour = range(7))
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Probability')
        
        filename = temp_folder + 'frame_{}.png'.format(i)
        filenames.append(filename)   
        
        plt.savefig(filename, dpi=50)
        plt.close()
        
        
    
    gif_name = 'pose_clustering_{}'.format(patient.patient_id) 
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
    os.rmdir(temp_folder)
    return
    
    
    
    
    
    