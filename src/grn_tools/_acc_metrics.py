import os
import sys
from tqdm.auto import tqdm
from rich.progress import Progress, track
from rich.console import Console

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, fbeta_score, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import matplotlib.pyplot as plt

from ._constants import DATASETS, GROUNDTRUTHS_TISSUE, DATA_DIR, GROUNDTRUTHS_LINEAGE, GROUNDTRUTHS_SCMULTISIM, NET_DIR_SCMULTISIM
from ._utils import matrix_to_edge, flatten, capitalize


__all__ = ["evaluate_TFbind", "batch_evaluate_TFbind", "batch_evaluate_TFbind_perTF", "batch_evaluate_scMultiSim"]

# ================ Metrics ================

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
        # If all predicted probabilities are 1, randomly select K edges as positive predictions
        if np.allclose(est["Score"], 1.0):
            print("All predicted probabilities are 1.0. Randomly selecting K edges as positive predictions.")
            select_k = min(K, len(y_probas))
            chosen_idx = np.random.choice(len(y_probas), size=select_k, replace=False)
            y_pred = np.zeros_like(y_probas, dtype=bool)
            y_pred[chosen_idx] = True
            _thres = 1.0
        else:
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

# ================ Evaluation Workflow =================

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
        aupr, _, _, _, _ = compute_AUPR(e_est, e_true, partial=0.3, plot=plot)
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
        aupr, _, _, _, _ = compute_AUPR(e_est, e_true, plot=plot)
        epr, _, _ = compute_EPR(e_est, e_true)
        fscore, precision, recall, thres = compute_Fscore(e_est, e_true, beta=1, thres_mode=thres_mode, plot=plot)

        print('AUPR: %f' % aupr)
        print('EPR: %f' % epr)
        print('F1score: %f' % fscore)
        

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

    console = Console()

    if mode == "Dictys":
        metrics = ["Partial AUPR", "F0.1 Score"]

    elif mode == "BEELINE": 
        metrics = ["EPR", "AUPR", "AUROC"]
        
    elif mode == "custom":
        metrics = ["AUPR", "AUROC", "F1 score(topk)", "EPR", "F0.1 score(topk)"]
        
    ds_list = datasets.keys()
    metrics_list = []

    with Progress(console=console) as progress:
        dataset_task = progress.add_task("[cyan]Processing datasets...", total=len(ds_list))

        for ds in ds_list:
            version = datasets[ds][0]
            lin_list = datasets[ds][1]
            grn_dir = data_dir + ds + "/benchmark/" + datasets[ds][0] + "/net/"

            lineage_task = progress.add_task(f"[cyan]Processing lineages for {ds}...", total=len(lin_list))

            for lin in lin_list:
                gt_ChIP = pd.read_csv(groundtruths[(ds, lin)], index_col=None)
                gt_ChIP.TF = gt_ChIP.TF.str.upper()
                gt_ChIP.Target = gt_ChIP.Target.str.upper()

                algo_task = progress.add_task(f"[cyan]Processing algorithms for {lin}...", total=len(algo_list))

                for algo in algo_list:
                    edge_est = pd.read_csv(grn_dir + f"{algo}_{lin}.csv", index_col=None).reset_index(drop=True)

                    edge_est.columns = ["TF", "Target", "Score"]
                    edge_est.TF = edge_est.TF.str.upper()
                    edge_est.Target = edge_est.Target.str.upper()
                    edge_est = edge_est[edge_est.Score >= 0]  # Just consider the positive values.

                    gt_GRN = gt_ChIP.loc[gt_ChIP.TF.isin(set(edge_est.TF)) & gt_ChIP.Target.isin(set(edge_est.Target))]
                    gt_GRN = gt_GRN[gt_GRN.TF != gt_GRN.Target]  # Remove self-loops.
                    gt_GRN.drop_duplicates(keep='first', inplace=True)  # Remove duplicates.
                    gt_GRN = gt_GRN[["TF", "Target"]]

                    if mode == "custom":
                        for metric in metrics:
                            if metric == "AUPR":
                                aupr, precs, recalls, auprr1, auprr2 = compute_AUPR(
                                    edge_est, gt_GRN, plot=plot, save=save, 
                                    save_prefix=f"{ds}#{algo}_{lin}#{version}"
                                )
                            elif metric == "EPR":
                                epr = compute_EPR(edge_est, gt_GRN)[0]
                            elif metric == "AUROC":
                                auroc, fprs, tprs = compute_AUROC(
                                    edge_est, gt_GRN, plot=plot, save=save, 
                                    save_prefix=f"{ds}#{algo}_{lin}#{version}"
                                )
                            elif metric == "F1 score(topk)":
                                f1, conf_mat, prec, recall, _ = compute_Fscore(
                                    edge_est, gt_GRN, beta=1, thres_mode='topk', plot=plot, save=save, 
                                    save_prefix=f"{ds}#{algo}_{lin}#{version}"
                                )
                            elif metric == "F0.1 score(topk)":
                                f01, conf_mat, prec, recall, _ = compute_Fscore(
                                    edge_est, gt_GRN, beta=0.1, thres_mode='topk', plot=plot, save=save, 
                                    save_prefix=f"{ds}#{algo}_{lin}#{version}"
                                )

                        n_tf = len(set(gt_GRN.TF))
                        n_target = len(set(gt_GRN.Target))
                        columns = [
                            "Algorithm", "Dataset", "Lineage", "Nums_TF", "Nums_Target", "Precsions", "Recalls", 
                            "AUPR", "AUPR Ratio1", "AUPR Ratio2", "EPR", "FPRs", "TPRs", "AUROC", 
                            "Confusion Matrix", "Precision", "Recall", "F1 Score(topk)", "F0.1 Score(topk)"
                        ]

                        metrics_list.append([
                            algo, ds, lin, n_tf, n_target, precs, recalls, aupr, auprr1, auprr2, 
                            epr, fprs, tprs, auroc, conf_mat, prec, recall, f1, f01
                        ])

                    progress.update(algo_task, advance=1)

                progress.update(lineage_task, advance=1)

            progress.update(dataset_task, advance=1)

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


def batch_evaluate_scMultiSim(algo_list, 
                              datasets=None, 
                              groundtruths=GROUNDTRUTHS_SCMULTISIM,
                              net_dir=NET_DIR_SCMULTISIM, 
                              mode="custom", 
                              plot=False, 
                              save=False, 
                              save_path=None):
    
    '''
    Batch evaluation of estimated metrics 
    
    Parameters:
    -----------
    algo_list: list of algorithms
    datasets: list of datasets
    groundtruths: dictionary of groundtruths 
    dataset_dir: base directory of data
    mode: evaluation metrics mode
    plot: plot evaluation results
    save: save evaluation results
    save_path: path to save evaluation results
    
    Return:
    -------
    
    metrics_dfs: evaluation results
    metrics_dict: evaluation results in dictionary format
    
    '''
    console = Console()
    
    if mode == "Dictys":
        metrics = ["Partial AUPR", "F0.1 Score"]

    elif mode == "BEELINE": 
        metrics = ["EPR", "AUPR", "AUROC"]
        
    elif mode == "custom":
        metrics = ["AUPR", "AUROC", "F1 score(topk)", "EPR", "F0.1 score(topk)"]
        
    metrics_list = []
    
    with Progress(console=console) as progress:
        
        algo_task = progress.add_task(f"[cyan]Processing algorithms...", total=len(algo_list))
        
        for algo in algo_list:
            net_algo_dir = os.path.join(net_dir, algo)
            
            if datasets is None:
                datasets = os.listdir(net_algo_dir)
                datasets = [ds for ds in datasets if ds.endswith(".csv")]
                datasets = [ds.split(".csv")[0] for ds in datasets]
                
            dataset_task = progress.add_task(f"[cyan]Processing datasets...", total=len(datasets))
            
            for ds in datasets:
                
                edge_est = pd.read_csv(os.path.join(net_algo_dir, f"{ds}.csv"), index_col=None).reset_index(drop=True)
                
                edge_est.columns = ["TF", "Target", "Score"]
                # edge_est.TF = edge_est.TF.str.upper()
                # edge_est.Target = edge_est.Target.str.upper()
                
                gt = ds.split("_")[0]
                gt_GRN = pd.read_csv(groundtruths[gt], index_col=None)
                
                gt_GRN = gt_GRN.iloc[:, 0:2]
                gt_GRN.columns = ["TF", "Target"]
                
                if len(edge_est) == 0:
                    print(f"Warning: {ds} has no edges.")
                    continue
                
                if mode == "custom":
                    for metric in metrics:
                        if metric == "AUPR":
                            aupr, precs, recalls, auprr1, auprr2   = compute_AUPR(
                                edge_est, gt_GRN, plot=plot, save=save, 
                                save_prefix=f"{ds}#{algo}"
                            )
                        # elif metric == "EPR":
                        #     epr = compute_EPR(edge_est, gt_GRN)[0]
                        elif metric == "AUROC":
                            auroc, fprs, tprs = compute_AUROC(
                                edge_est, gt_GRN, plot=plot, save=save, 
                                save_prefix=f"{ds}#{algo}"
                            )
                        elif metric == "F1 score(topk)":
                            f1, conf_mat, prec, recall, _ = compute_Fscore(
                                edge_est, gt_GRN, beta=1, thres_mode='topk_perTF', plot=plot, save=save, 
                                save_prefix=f"{ds}#{algo}"
                            )
                        elif metric == "F0.1 score(topk)":
                            f01, conf_mat, prec, recall, _ = compute_Fscore(
                                edge_est, gt_GRN, beta=0.1, thres_mode='topk_perTF', plot=plot, save=save, 
                                save_prefix=f"{ds}#{algo}"
                            )
                    n_tf = len(set(gt_GRN.TF))
                    n_target = len(set(gt_GRN.Target))
                    
                    columns = [
                            "Algorithm", "Dataset", "Nums_TF", "Nums_Target", "Precsions", "Recalls", 
                            "AUPR", "AUPR Ratio1", "AUPR Ratio2", "FPRs", "TPRs", "AUROC", 
                            "Confusion Matrix", "Precision", "Recall", "F1 Score(topk)", "F0.1 Score(topk)"
                        ]
                    
                    metrics_list.append([
                        algo, ds, n_tf, n_target, precs, recalls, aupr, auprr1, auprr2, 
                        fprs, tprs, auroc, conf_mat, prec, recall, f1, f01
                    ])
                    
                progress.update(dataset_task, advance=1)
            
            progress.update(algo_task, advance=1)
                
        metrics_df = pd.DataFrame(metrics_list, columns=columns)
        
        if save:
            metrics_df.to_csv(save_path)
        
        return metrics_df
            
            
            
            
            
        
        
        
        
        
    

