import numpy as np
import pandas as pd

import matplotlib
import mplscience
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
matplotlib.rcParams["figure.figsize"] = [4, 4]
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["image.cmap"] = "Spectral_r"

import os,sys
from copy import deepcopy
from itertools import product, permutations
from scipy.stats import hypergeom

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, fbeta_score, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

#### Constants ####
DATA_DIR = "/home/chenxufeng/picb_cxf/Data/"

DATASETS = {"PMID36973557_NatBiotechnol2023_CD34":['240921',['Ery', 'Mono', 'CLP', 'Mega', 'cDC', 'pDC']],
            "PMID36973557_NatBiotechnol2023_T-cell-depleted": ['240704', ['Ery', 'Mono', 'NaiveB']],
            "bioRxiv_Klein2023_Pancreas": ['240914', ['Alpha', 'Beta', 'Delta', 'Epsilon']]}

GT_DIR = "/home/chenxufeng/picb_cxf/Database/CistromeDB/"

GROUNDTRUTHS_LINEAGE = {
    ("PMID36973557_NatBiotechnol2023_CD34", "Ery"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Ery_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mono"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mono_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "CLP"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_CLP_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mega"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mega_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "cDC"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_cDC_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "pDC"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_pDC_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Ery"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Ery_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Mono"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mono_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "NaiveB"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_NaiveB_top1000_network.csv"
}

GROUNDTRUTHS_TISSUE = {
    ("PMID36973557_NatBiotechnol2023_CD34", "Ery"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mono"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "CLP"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mega"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "cDC"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "pDC"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Ery"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Mono"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "NaiveB"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Alpha"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Beta"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Delta"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Epsilon"): "/home/chenxufeng/picb_cxf/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv"
}

#### Helper functions ####

def flatten(xss):
    '''
    Flatten a list of lists
    
    Input:
    ------
    xss: list of lists
    
    Return: 
    ------
    flattened list
    '''
    return np.array([x for xs in xss for x in xs])

def capitalize(s):
    '''
    Capitalize a string
    
    Input:
    ------
    s: string
    
    Return:
    -------
    capitalized string
    '''
    return s[0].upper() + s[1:]

def matrix_to_edge(m, rownames, colnames):
    '''
    Convert matrix to edge list
    p.s. row for regulator, column for target
    
    
    Parameters:
    -----------
    m: matrix
    rownames: list of regulator names
    colNames: list of target names

    Return:
    -------
    edge DataFrame [TF, Target, Score]
    '''
    
    mat = deepcopy(m)
    mat = pd.DataFrame(mat)

    rownames = np.array(rownames)
    colnames = np.array(colnames)
    
    num_regs = rownames.shape[0]
    num_targets = colnames.shape[0]

    mat_indicator_all = np.zeros([num_regs, num_targets])

    mat_indicator_all[abs(mat) > 0] = 1
    idx_row, idx_col = np.where(mat_indicator_all)

    idx = list(zip(idx_row, idx_col))
    #for row, col in idx:
    #    if row == col:
    #        idx.remove((row, col))
                
    edges_df = pd.DataFrame(
        {'TF': rownames[idx_row], 'Target': colnames[idx_col], 'Score': [mat.iloc[row, col] for row, col in idx]})

    edge = edges_df.sort_values('Score', ascending=False)
    return edge

### Metrics

def compute_EPR(est, true):
    '''
    Compute early precision rate (EPR)
    
    Formula: EPR = EP / RP
    EP: early precision
    RP: random precision
    
    Parameters:
    -----------
    e_est: estimated edge list
    e_true: true edge list

    Return: 
    -------
    EPR, EP, RP
    '''

    est = est.copy()
    true = true.copy()

    est.columns = ['TF', 'Target', 'Score']
    true.columns = ['TF', 'Target']


    est['Score'] = abs(est['Score'])
    est = est.sort_values('Score', ascending=False)

    #e_est = e_est.astype(int)
    #e_true = e_true.astype(int)

    reg = set(true['TF'])
    gene = set(true['TF']) | set(true['Target'])

    est = est[est['TF'].apply(lambda x: x in reg)]
    est = est[est['Target'].apply(lambda x: x in gene)]

    est = est.astype('str')
    true = true.astype('str')

    true_set = set(true['TF']+'|'+true['Target'])

    est = est.iloc[:len(true_set)]

    est_set = set(est['TF']+'|'+est['Target'])
    
    # Random predictor's precision
    regnums = len(reg)
    genenums = len(gene)
    rp = len(true_set)**2/(regnums*(genenums-1))

    # Early precision
    ep = len(est_set & true_set)
 
    epr = ep / rp
    return epr, ep, rp

def compute_AUPR(est, true, partial=1, plot=False, save=False, save_prefix=None):
    """
    Compute area under precision-recall curve (AUPR)
    
    Parameters:
    -----------
    est: estimated edge dataframe [TF, Target, Score]
    true: true edge dataframe [TF, Target]
    partial: partial AUPR, default 1
    plot: plot AUPR curve
    save: save plot
    save_prefix: prefix to save plot
    
    Return:
    -------
    AUPR, precision, recall
    """

    est = est.copy()
    true = true.copy()

    est.columns = ['TF', 'Target', 'Score']
    true.columns = ['TF', 'Target']

    est['Score'] = abs(est['Score'])
    est = est.sort_values('Score',ascending=False)

    TFs = set(true['TF'])
    Genes = set(true['TF'])| set(true['Target'])

    est = est[est['TF'].apply(lambda x: x in TFs)]
    est = est[est['Target'].apply(lambda x: x in Genes)]
    
    res_d = {}
    y_true = []
    y_probas= []

    for item in (est.to_dict('records')):
            res_d[item['TF']+item['Target']] = item['Score']
    true_set = set(true['TF']+true['Target'])

    for item in (set(true['TF'])):
            for item2 in set(true['TF'])| set(true['Target']):
                if item+item2 in true_set:
                    y_true.append(1)
                else:
                    y_true.append(0)
                if item+item2 in res_d:
                    y_probas.append(res_d[item+item2])
                else:
                    y_probas.append(0)

    precision, recall, thres = precision_recall_curve(y_true, y_probas)
    precision = precision[recall <= partial]
    recall = recall[recall <= partial]
    
    aupr = average_precision_score(y_true,y_probas)
    #aupr = auc(recall, precision)
    
    ## AUPR ratio
    auprr1 = aupr*len(y_true)/sum(y_true)
    
    
    # Calculate AUPR for a random predictor
    random_probas = np.random.rand(len(y_true))
    random_aupr = average_precision_score(y_true, random_probas)
    
    # Calculate AUPR ratio
    auprr2 = aupr / random_aupr
    
    if plot: 
        plt.figure(figsize=(5, 4))
        precision, recall, thres = precision_recall_curve(y_true, y_probas)
        plt.title('Precision-Recall Curve')
        plt.plot(recall, precision, 'b', label = 'AUPRC = %0.3f' % aupr)
        plt.legend(loc = 'lower right')
        plt.xlim([-0.05, partial])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        
        if save:
            data_dir = "/home/chenxufeng/picb_cxf/Data/"
            plt.savefig(os.path.join(data_dir, 
                                    save_prefix.split("#")[0], 
                                    "benchmark", 
                                    save_prefix.split("#")[2], 
                                    "fig", 
                                    f'{save_prefix.split("#")[1]}_AUPR.pdf'), bbox_inches='tight')
            plt.close()

        plt.show()

    return aupr, precision, recall, auprr1, auprr2


def compute_AUROC(est, true, plot=False, save=False, save_prefix=None):
    '''
    Compute area under ROC curve (AUROC)
    
    Parameters:
    -----------
    est: estimated edge dataframe [TF, Target, Score]
    true: true edge dataframe [TF, Target]
    plot: plot ROC curve
    save: save plot
    save_prefix: prefix to save plot
    
    Return: 
    -------
    AUROC, fpr, tpr
    '''

    est = est.copy()
    true = true.copy()

    est.columns = ['TF', 'Target', 'Score']
    true.columns = ['TF', 'Target']

    est['Score'] = abs(est['Score'])
    est = est.sort_values('Score',ascending=False)

    TFs = set(true['TF'])
    Genes = set(true['TF'])| set(true['Target'])

    est = est[est['TF'].apply(lambda x: x in TFs)]
    est = est[est['Target'].apply(lambda x: x in Genes)]
    
    res_d = {}
    y_true = []
    y_probas= []

    for item in (est.to_dict('records')):
            res_d[item['TF']+item['Target']] = item['Score']
    true_set = set(true['TF']+true['Target'])

    for item in (set(true['TF'])):
            for item2 in set(true['TF'])| set(true['Target']):
                if item+item2 in true_set:
                    y_true.append(1)
                else:
                    y_true.append(0)
                if item+item2 in res_d:
                    y_probas.append(res_d[item+item2])
                else:
                    y_probas.append(0)

    
    auroc = roc_auc_score(y_true,y_probas)
    fpr, tpr, _ = roc_curve(y_true, y_probas)

    if plot:
        plt.figure(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_true, y_probas)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUROC = %0.3f' % auroc)
        plt.legend(loc = 'lower right')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        
        if save:
            data_dir = "/home/chenxufeng/picb_cxf/Data/"
            plt.savefig(os.path.join(data_dir, 
                                    save_prefix.split("#")[0], 
                                    "benchmark", 
                                    save_prefix.split("#")[2], 
                                    "fig", 
                                    f'{save_prefix.split("#")[1]}_AUROC.pdf'), bbox_inches='tight')
            plt.close()

        plt.show()

    return auroc, fpr, tpr

def compute_Fscore(est, 
                   true, 
                   beta=1, 
                   thres_mode='max', 
                   plot=False,
                   save=False,
                   save_prefix=None):
    '''   
    Compute F-beta score
    
    Parameters:
    -----------
    est: estimated edge dataframe [TF, Target, Score]
    true: true edge dataframe [TF, Target]
    beta: beta value
    thres_mode: threshold mode
    plot: plot evaluation results
    save: save evaluation results
    save_prefix: prefix to save evaluation results
    
    Return:
    -------
    F-beta score, confusion matrix, precision, recall, threshold
    '''
    est = est.copy()
    true = true.copy()

    est.columns = ['TF', 'Target', 'Score']
    true.columns = ['TF', 'Target']

    est['Score'] = abs(est['Score'])
    est = est.sort_values('Score',ascending=False)

    TFs = set(true['TF'])
    Genes = set(true['TF'])| set(true['Target'])

    est = est[est['TF'].apply(lambda x: x in TFs)]
    est = est[est['Target'].apply(lambda x: x in Genes)]
    
    res_d = {}
    y_true = []
    y_probas= []

    for item in (est.to_dict('records')):
            res_d[item['TF']+item['Target']] = item['Score']
    true_set = set(true['TF']+true['Target'])

    for item in (set(true['TF'])):
            for item2 in set(true['TF'])| set(true['Target']):
                if item+item2 in true_set:
                    y_true.append(1)
                else:
                    y_true.append(0)
                if item+item2 in res_d:
                    y_probas.append(res_d[item+item2])
                else:
                    y_probas.append(0)
    
    
    # y_pred = np.zeros_like(y_probas)
    # The F0.1 scores for continuous GRNs were computed as its largest value across all possible cutoffs that converts a continuous GRN to a binary GRN.
    beta = beta
    y_probas = np.array(y_probas)
    y_true = np.array(y_true)
    
    
    if thres_mode == 'max':
    
        precision, recall, thres = precision_recall_curve(y_true, y_probas)
        positive_indices = thres > 0.0
        precision = precision[:-1][positive_indices]
        recall = recall[:-1][positive_indices]
        thres = thres[positive_indices]
        
        fscore = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        fscore[np.isnan(fscore)] = 0.0
        max_index = fscore.argmax()
        _fscore = fscore[max_index]
        _prec = precision[max_index]
        _recall = recall[max_index]
        _thres = thres[max_index]
        _conf_mat = confusion_matrix(y_true, y_pred)
    
    elif thres_mode == 'topk':
        K = len(true_set)
        _thres = np.sort(y_probas)[-K]
        if _thres == 0.0:
            # print("True: ", np.count_nonzero(y_true), "Estimate:", np.count_nonzero(y_probas))
            _thres = np.min(y_probas[y_probas > 0.0])
            # print(f"Warning: The threshold is 0.0. Setting the threshold to the minimum non-zero value {_thres}")
        y_pred = np.array(y_probas) >= _thres
        _fscore = fbeta_score(y_true, y_pred, beta=beta, average="binary")
        _prec = precision_score(y_true, y_pred, average="binary")
        _recall = recall_score(y_true, y_pred, average="binary")
        _conf_mat = confusion_matrix(y_true, y_pred)
        
    elif thres_mode == 'topk_perTF':
        _thres = 0
        y_pred = np.array(y_probas) > _thres
        _fscore = fbeta_score(y_true, y_pred, beta=beta, average="binary")
        _prec = precision_score(y_true, y_pred, average="binary")
        _recall = recall_score(y_true, y_pred, average="binary")
        _conf_mat = confusion_matrix(y_true, y_pred)
        
    elif thres_mode == 'quantile':
        pos_percentile = np.percentile(y_true, 100)
        print(np.count_nonzero(y_true), np.count_nonzero(y_probas))
        print(pos_percentile)
        _thres = np.percentile(y_probas, pos_percentile)
        if _thres == 0.0:
            print(np.count_nonzero(y_true), np.count_nonzero(y_probas))
            # print("Warning: The threshold is 0.0. Setting the threshold to the minimum non-zero value.")
            _thres = np.min(y_probas[y_probas > 0.0])
        y_pred = np.array(y_probas) >= _thres
        _fscore = fbeta_score(y_true, y_pred, beta=beta, average="binary")
        _prec = precision_score(y_true, y_pred, average="binary")
        _recall = recall_score(y_true, y_pred, average="binary")
        _conf_mat = confusion_matrix(y_true, y_pred)
        
    if plot:
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot histogram of y_probas
        y_probas_filtered = [prob for prob in y_probas if prob != 0.0]
        axs[0].hist(y_probas_filtered, bins=50, alpha=0.7)
        axs[0].axvline(x=_thres, color='r', linestyle='--', label=f'Best Threshold: {_thres:.2f}\nF Score: {_fscore:.2f}\nTF: {len(TFs)}\nTarget: {len(Genes)}\nEdges: {len(y_true)}')
        
        axs[0].set_xlabel('Regulation strength')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Distribution of Regulation strength')
        axs[0].legend()
        
        # Confusion matrix after applying the threshold
        y_pred = np.array(y_probas) >= _thres
        cm = confusion_matrix(y_true, y_pred)
        im = axs[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.7)
        axs[1].set_title('Confusion Matrix')
        fig.colorbar(im, ax=axs[1])

        # Add labels to each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1].text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
                
        # Add TP, FP, TN, FN labels
        axs[1].text(-0.3, 0, 'TN', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
        axs[1].text(0.7, 0, 'FP', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
        axs[1].text(-0.3, 1, 'FN', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
        axs[1].text(0.7, 1, 'TP', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
        
        # Remove axis ticks
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        # Add a suptitle
        fig.suptitle(f'{save_prefix}', fontsize=16)
        
        # Save the figure as a PDF
        if save:
            data_dir = "/home/chenxufeng/picb_cxf/Data/"
            plt.savefig(os.path.join(data_dir, 
                                    save_prefix.split("#")[0], 
                                    "benchmark", 
                                    save_prefix.split("#")[2], 
                                    "fig", 
                                    f'{save_prefix.split("#")[1]}_Fscore.pdf'), bbox_inches='tight')
            plt.close()

        plt.show()
        # plt.tight_layout()
        # plt.show()

    # fscore = fbeta_score(y_true, y_pred, beta=beta, average=average)
    return _fscore, _conf_mat, _prec, _recall, _thres


def evaluate_TFbind(e_est, e_true, mode=None, thres_mode="max", plot=False):

    '''
    Evaluate estimated metrics
    
    Parameters:
    -----------
    e_est: estimated edge dataframe [TF, Target, Score]
    e_true: true edge dataframe [TF, Target]
    mode: evaluation mode
    thres_mode: threshold mode
    plot: plot evaluation results
    
    Return:
    -------
    evaluation results
    '''
    if mode == "Dictys":
        aupr, _, _ = compute_AUPR(e_est, e_true, partial=0.3, plot=plot)
        fscore, precision, recall, thres = compute_Fscore(e_est, e_true, beta=0.1, thres_mode=thres_mode, plot=plot)

        print('Partial AUPR: %f' % aupr)
        print('F0.1 Score: %f Precision: %f Recall: %f with threshold: %f' % (fscore, precision, recall, thres))

    if mode == "BEELINE":
        epr, ep, rp = compute_EPR(e_est, e_true)
        aupr, precision, recall = compute_AUPR(e_est, e_true, plot=plot)
        auroc, fpr, tpr = compute_AUROC(e_est, e_true, plot=plot)
        #fscore = compute_Fscore(e_est, e_true, beta=1)

        print('EPR: %f, EP: %f, RP: %f' % (epr, ep, rp))
        print('AUPR: %f' % aupr)
        print('AUROC: %f' % auroc)
        #print('F1score: %f' % fscore)
    
    if mode == "custom":
        aupr, _, _ = compute_AUPR(e_est, e_true, plot=plot)
        epr, _, _ = compute_EPR(e_est, e_true)
        fscore, precision, recall, thres = compute_Fscore(e_est, e_true, beta=1, thres_mode=thres_mode, plot=plot)

        print('AUPR: %f' % aupr)
        print('EPR: %f' % epr)
        print('F1score: %f' % fscore)
        
            
### Pipeline
        
def batch_evaluate_TFbind(algo_list, 
                            datasets=DATASETS, 
                            groundtruths=GROUNDTRUTHS_TISSUE,
                            data_dir=DATA_DIR, 
                            mode="custom", 
                            plot=False, 
                            save=False, 
                            save_path=None):
    '''
    Batch evaluation of estimated metrics
    
    Parameters:
    -----------
    algo_list: list of algorithms
    datasets: dictionary of datasets {dataset_name: (version, [lineages])}
    data_dir: base directory of data
    mode: evaluation metrics mode
    chip: ChIP-seq data type
    top_peak: number of top peaks
    plot: plot evaluation results
    save: save evaluation results
    save_path: path to save evaluation results
    
    Return:
    -------
    metrics_dfs: evaluation results
    metrics_dict: evaluation results in dictionary format
    
    '''
    if mode == "Dictys":
        metrics = ["Partial AUPR", "F0.1 Score"]

    elif mode == "BEELINE": 
        metrics = ["EPR", "AUPR", "AUROC"]
        
    elif mode == "custom":
        metrics = ["AUPR", "AUROC", "F1 score(topk)", "EPR", "F0.1 score(topk)"]
        
    ds_list = datasets.keys()
    
    metrics_list = []
    for ds in ds_list:
        print(f"-----------------------{ds}--------------------------")
        version = datasets[ds][0]
        lin_list = datasets[ds][1]
        grn_dir = data_dir + ds + "/benchmark/" + datasets[ds][0] + "/net/"
        

        for lin in lin_list:
            gt_ChIP = pd.read_csv(groundtruths[(ds, lin)], index_col=None)
            gt_ChIP.TF = gt_ChIP.TF.str.upper()
            gt_ChIP.Target = gt_ChIP.Target.str.upper()
            
            for algo in tqdm(algo_list, desc=f"{ds}#{lin}"):
                edge_est = pd.read_csv(grn_dir + f"{algo}_{lin}.csv", index_col=None).reset_index(drop=True)
                
                edge_est.columns = ["TF", "Target", "Score"]
                # edge_est.Score = edge_est.Score.abs()
                edge_est.TF = edge_est.TF.str.upper()
                edge_est.Target = edge_est.Target.str.upper()
                # Just consider the positive values.
                edge_est = edge_est[edge_est.Score >= 0]

                gt_GRN = gt_ChIP.loc[gt_ChIP.TF.isin(set(edge_est.TF)) & gt_ChIP.Target.isin(set(edge_est.Target))]
                # Remove self-loops.
                gt_GRN  = gt_GRN[gt_GRN.TF != gt_GRN.Target]
                # Remove duplicates (there are some repeated lines in the ground-truth networks!!!). 
                gt_GRN.drop_duplicates(keep = 'first', inplace=True)

                gt_GRN = gt_GRN[["TF", "Target"]]

                if mode == "custom":
                    for metric in metrics:
                        if metric == "AUPR":
                            aupr,fprs,tprs,auprr1,auprr2 = compute_AUPR(edge_est, gt_GRN, plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                            
                        elif metric == "EPR":
                            epr = compute_EPR(edge_est, gt_GRN)[0]
                            
                        elif metric == "AUROC":
                            auroc,precs,recalls = compute_AUROC(edge_est, gt_GRN, plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                        
                        elif metric == "F1 score(topk)":
                            f1,conf_mat,prec,recall,_ = compute_Fscore(edge_est, gt_GRN, beta=1, thres_mode='topk', plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                            
                        elif metric == "F0.1 score(topk)":
                            f01,conf_mat,prec,recall,_ = compute_Fscore(edge_est, gt_GRN, beta=0.1, thres_mode='topk', plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                            
                    n_tf = len(set(gt_GRN.TF))
                    n_target = len(set(gt_GRN.Target))
                    columns = ["Algorithm", "Dataset", "Lineage", "Nums_TF", "Nums_Target", "Precsions", "Recalls", "AUPR", "AUPR Ratio1", "AUPR Ratio2", "EPR", "FPRs", "TPRs", "AUROC", "Confusion Matrix", "Precision", "Recall", "F1 Score(topk)", "F0.1 Score(topk)"]
                    metrics_list.append([algo, ds, lin, n_tf, n_target, precs, recalls, aupr, auprr1, auprr2, epr, fprs, tprs, auroc, conf_mat, prec, recall, f1, f01])
        
    metrics_df = pd.DataFrame(metrics_list, columns=columns)

    if save:
        metrics_df.to_csv(save_path)

    return metrics_df

            
def batch_evaluate_TFbind_perTF(algo_list, 
                                 datasets=DATASETS, 
                                 groundtruths=GROUNDTRUTHS_TISSUE,
                                 data_dir=DATA_DIR, 
                                 mode="custom", 
                                 plot=False, 
                                 save=False, 
                                 save_path=None):
    '''
    Batch evaluation of estimated metrics
    
    Parameters:
    -----------
    algo_list: list of algorithms
    datasets: dictionary of datasets {dataset_name: (version, [lineages])}
    data_dir: base directory of data
    mode: evaluation metrics mode
    chip: ChIP-seq data type
    top_peak: number of top peaks
    plot: plot evaluation results
    save: save evaluation results
    save_path: path to save evaluation results
    
    Return:
    -------
    metrics_dfs: evaluation results
    metrics_dict: evaluation results in dictionary format
    
    '''
    if mode == "Dictys":
        metrics = ["Partial AUPR", "F0.1 Score"]

    elif mode == "BEELINE": 
        metrics = ["EPR", "AUPR", "AUROC"]
        
    elif mode == "custom":
        metrics = ["AUPR", "AUROC", "F1 score(topk)", "EPR", "F0.1 score(topk)"]
        
    ds_list = datasets.keys()
    
    
    metrics_list = []
    for ds in ds_list:
        print(f"-----------------------{ds}--------------------------")
        version = datasets[ds][0]
        lin_list = datasets[ds][1]
        grn_dir = data_dir + ds + "/benchmark/" + datasets[ds][0] + "/net/"
        
        for lin in lin_list:
            gt_ChIP = pd.read_csv(groundtruths[(ds, lin)], index_col=None)
            gt_ChIP.TF = gt_ChIP.TF.str.upper()
            gt_ChIP.Target = gt_ChIP.Target.str.upper()
            
            for algo in algo_list:
                # print(f"Processing {algo} on {lin} lineage")
                edge_est = pd.read_csv(grn_dir + f"{algo}_{lin}.csv", index_col=None).reset_index(drop=True)
                
                edge_est.columns = ["TF", "Target", "Score"]
                # edge_est.Score = edge_est.Score.abs()
                edge_est.TF = edge_est.TF.str.upper()
                edge_est.Target = edge_est.Target.str.upper()
                # Just consider the positive values.
                edge_est = edge_est[edge_est.Score >= 0]

                gt_GRN = gt_ChIP.loc[gt_ChIP.TF.isin(set(edge_est.TF)) & gt_ChIP.Target.isin(set(edge_est.Target))]
                # Remove self-loops.
                gt_GRN  = gt_GRN[gt_GRN.TF != gt_GRN.Target]
                # Remove duplicates (there are some repeated lines in the ground-truth networks!!!). 
                gt_GRN.drop_duplicates(keep = 'first', inplace=True)

                gt_GRN = gt_GRN[["TF", "Target"]]
                
                tf_list = set(gt_GRN.TF)
                
                # Select TopK edges
                K = len(gt_GRN)
                
                if len(edge_est) < K:
                    threshold = np.min(edge_est.Score)
                else:
                    threshold = np.sort(edge_est.Score)[-K]
                edge_est_topk = edge_est.copy()
                edge_est_topk.Score = edge_est.Score.apply(lambda x: x if x >= threshold else 0)
                
                for tf in tf_list:
                    
                    print(f"Processing {algo} on {lin} lineage for {tf}")
                    edge_est_tf = edge_est[edge_est.TF == tf]
                    edge_est_tf_topk = edge_est_topk[edge_est_topk.TF == tf]

                    if len(edge_est_tf_topk) == 0:
                        continue
                    
                    gt_GRN_tf = gt_GRN[gt_GRN.TF == tf]

                    if mode == "custom":
                        for metric in metrics:
                            if metric == "AUPR":
                                aupr,fprs,tprs,auprr1,auprr2 = compute_AUPR(edge_est, gt_GRN, plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                                
                            elif metric == "EPR":
                                epr = compute_EPR(edge_est_tf, gt_GRN_tf)[0]
                                
                            elif metric == "AUROC":
                                auroc,precs,recalls = compute_AUROC(edge_est, gt_GRN, plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                            
                            elif metric == "F1 score(topk)":
                                f1,conf_mat,prec,recall,_ = compute_Fscore(edge_est_tf_topk, gt_GRN_tf, beta=1, thres_mode='topk_perTF', plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")
                            
                            elif metric == "F0.1 score(topk)":
                                f01,conf_mat,prec,recall,_ = compute_Fscore(edge_est_tf_topk, gt_GRN_tf, beta=0.1, thres_mode='topk_perTF', plot=plot, save=save, save_prefix=f"{ds}#{algo}_{lin}#{version}")        
                        
                        
                        n_target = len(set(gt_GRN_tf.Target))
                        columns = ["Algorithm", "Dataset", "Lineage", "TF", "Nums_Target", "Precsions", "Recalls", "AUPR", "AUPR Ratio1", "AUPR Ratio2", "EPR", "FPRs", "TPRs", "AUROC", "Confusion Matrix", "Precision", "Recall", "F1 Score(topk)", "F0.1 Score(topk)"]
                        metrics_list.append([algo, ds, lin, tf, n_target, precs, recalls, aupr, auprr1, auprr2, epr, fprs, tprs, auroc, conf_mat, prec, recall, f1, f01])
        
    metrics_df = pd.DataFrame(metrics_list, columns=columns)

    if save:
        metrics_df.to_csv(save_path)

    return metrics_df

### Plotting functions ###

def plot_metrics(dfs, algo_list):
    
    dfs = {metric: dfs[metric].loc[dfs[metric].Algorithm.isin(algo_list) & dfs[metric].Lineage.isin(lin_list)] for metric in dfs.keys()}
    n_algo = len(algo_list)
    n_metrics = len(dfs)

    # palette = dict(zip(algo_list, sns.color_palette("colorblind").as_hex()[:n_algo]))
    palette = dict(zip(algo_list, [f"C{i}" for i in range(n_algo)]))

    with mplscience.style_context():
        fig, ax = plt.subplots(figsize=(10 * n_metrics, 6), ncols=n_metrics)

        for ax_id, metric in enumerate(dfs.keys()):
            _df = dfs[metric]
            _df["x_combined"] = _df["Lineage"] + "\n" + _df["Dataset"]
            sns.barplot(data=_df,
                        x="x_combined",
                        y="metric",
                        hue="Algorithm",
                        palette=palette,
                        ax=ax[ax_id],
                        )
            
            #ax[ax_id].set_title(metric)
            if ax_id == 0:
                ax[ax_id].set_ylabel(metric)
                handles, labels = ax[ax_id].get_legend_handles_labels()
            #ax[ax_id].set_xlabel("Lineage")
            ax[ax_id].text(0, -0.03, "PMID36973557_NatBiotechnol2023_CD34", )
            ax[ax_id].text(2.6, -0.03, "PMID36973557_NatBiotechnol2023_T-cell-depleted", )
            
            ax[ax_id].get_legend().remove()
            ax[ax_id].set_ylabel(metric)
            ax[ax_id].set_xticklabels(['Ery', 'Mono', 'CLP', 'Ery', 'Mono', 'NaiveB'])
            ax[ax_id].set_xlabel(None)
            
            #ax[ax_id].set_xticklabels(ax[ax_id].get_xticklabels(), rotation=45, ha='right')
        fig.legend(handles=handles, labels=labels, loc="lower center", ncol=n_algo, bbox_to_anchor=(0.5, -0.06))
        plt.tight_layout()

def plot_precision_recall(dfs, algo_list, lin_list):

    dfs = {lin: dfs[lin].loc[dfs[lin].Algorithm.isin(algo_list)] for lin in lin_list}

    n_algo = len(algo_list)
    n_lin = len(lin_list)

    palette = dict(zip(algo_list, sns.color_palette("colorblind").as_hex()[:n_algo]))

    with mplscience.style_context():
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(6 * n_lin, 4), ncols=n_lin)

        for ax_id, lin in enumerate(lin_list):
            _df = dfs[lin]
            sns.lineplot(data=_df, 
                         x="Recall", 
                         y="Precision", 
                         hue="Algorithm", 
                         ax=ax[ax_id]
                         )
            
            ax[ax_id].set_title(lin)
            if ax_id == 0:
                ax[ax_id].set_ylabel("Precision")
                handles, labels = ax[ax_id].get_legend_handles_labels()
            ax[ax_id].set_xlabel("Recall")
            ax[ax_id].get_legend().remove()
            ax[ax_id].set_ylim(-0.05, 0.4)
            ax[ax_id].set_xlim(-0.05, 0.4)

    #handles = [handles[0], handles[1], handles[2], handles[5], handles[4], handles[3]]
    #labels = [labels[0], labels[1], labels[2], labels[5], labels[4], labels[3]]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=n_algo, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.show()
    
### Others ### 
     
def dictys_binlinking2edge(file):
    prior_GRN = pd.read_csv(file, sep='\t', index_col=0)
    prior_GRN = prior_GRN.groupby(level=0).max()
    prior_GRN = prior_GRN.astype(int).copy()
    
    edge_list = matrix_to_edge(prior_GRN, prior_GRN.index, prior_GRN.columns)
    
    return edge_list






















