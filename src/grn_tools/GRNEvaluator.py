import os
import re
from itertools import combinations
from typing import Dict, List, Tuple, Union, Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns

# --- Rich for enhanced console output ---
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# --- From grn_tools package ---
# Ensure the grn_tools package is in your Python path
from grn_tools._acc_metrics import (compute_AUPR, compute_AUROC, compute_EPR,
                                    compute_Fscore)
from grn_tools._constants import (DATASETS, GROUNDTRUTHS_TISSUE,
                                  METHOD_PALETTE_SCMULTISIM, METHOD_PALETTE_CISTROME)
from grn_tools._plotting import plot_horizontal_boxplot, plot_scatter_with_error_bars, plot_violin
from grn_tools._stab_metrics import cosine_similarity, jaccard_similarity

import scmagnify as scm
scm_dir = os.path.dirname(scm.__file__)
human_tf_file = os.path.join(scm_dir, "data", "tf_lists", f"allTFs_hg38.txt")


class GRNEvaluator:
    """
    An integrated class for comprehensive Gene Regulatory Network (GRN) evaluation.

    This class provides a full suite of tools to manage, evaluate, and visualize
    the performance of GRN inference algorithms. It can handle loading networks from
    various sources, comparing them against ground truths, calculating accuracy and
    stability metrics, and generating a wide range of plots for insightful analysis.
    """

    def __init__(self):
        """Initializes the Evaluator instance."""
        self.networks = []  # Stores loaded GRNs, each as a dict with 'meta' and 'data'
        self.groundtruths = {}  # Stores ground truth networks, keyed by (dataset, lineage)
        self.grn_info = None  # Caches the description table for predicted networks
        self.groundtruth_info = None # Caches the description table for ground truths
        self.accuracy_results = None  # DataFrame storing summary accuracy metrics
        self.accuracy_details = []  # List of dicts with detailed results, including curve data
        self.stability_results = {}  # Stores stability metric results (e.g., Jaccard, Cosine)
        self.console = Console()  # Rich console for pretty printing
        self.algo_palette = METHOD_PALETTE_CISTROME  # Default color palette for algorithms
        self.gene_filter = []


    def __repr__(self):
        """String representation of the Evaluator instance."""
        return f"<GRNEvaluator: {len(self.networks)} networks loaded, {len(self.groundtruths)} ground truths>"

    def load_grns(
        self,
        source: Union[pd.DataFrame, str],
        algo: str = None,
        dataset: str = None,
        lineage: str = None,
        regex: str = None,
        tf_filter: Optional[List[str]] = None,
        describe: bool = False,
        **kwargs,
    ):
        """
        Adds predicted networks from various sources like DataFrames, file paths, or directories.

        Parameters:
        -----------
        source (Union[pd.DataFrame, str]): The data source. Can be:
            - A pandas DataFrame: Adds a single network. `algo`, `dataset`, and `lineage` are required.
            - A file path (str): Adds a single network from a CSV file. `algo`, `dataset`, and `lineage` are required.
            - A directory path (str): Scans and adds all matching networks. `dataset` is required.
        algo (str, optional): The name of the algorithm. Required for DataFrame/file sources.
        dataset (str, optional): The name of the dataset. Required for all source types.
        lineage (str, optional): The cell lineage. Required for DataFrame/file sources.
        regex (str, optional): A regex pattern to filter filenames when loading from a directory.
        describe (bool): If True, prints the network description table after loading.
        **kwargs: Additional custom metadata to be stored with each network.

        Return:
        -------
        None
        """
        # --- Case 1: Source is a Directory Path ---
        if isinstance(source, str) and os.path.isdir(source):
            if not dataset:
                self.console.print("[red]Error: 'dataset' must be provided when loading from a directory.[/]")
                return

            directory_path = source
            try:
                filenames = os.listdir(directory_path)
            except FileNotFoundError:
                self.console.print(f"[red]Error: Directory not found at '{directory_path}'.[/]")
                return
            
            # --- **NEW**: Pre-filter filenames using the provided regex pattern ---
            if regex:
                try:
                    user_pattern = re.compile(regex)
                    original_count = len(filenames)
                    filenames = [f for f in filenames if user_pattern.search(f)]
                    self.console.print(f"🔍 Regex filter '{regex}' matched {len(filenames)} out of {original_count} files.")
                except re.error as e:
                    self.console.print(f"[red]Error: Invalid regex pattern '{regex}': {e}[/]")
                    return
            
            # Define the regex pattern to extract algorithm and lineage from filenames
            parsing_pattern = re.compile(r"(.+)_([^_]+)\.csv$")
            loaded_count = 0
            
            for filename in track(filenames, description=f"⚙️[bold cyan]Scanning '{dataset}'..."):
                match = parsing_pattern.match(filename)
                if match:
                    parsed_algo, parsed_lineage = match.groups()
                    file_path = os.path.join(directory_path, filename)
                    # Use the internal helper to add the network
                    self._add_single_network(file_path, parsed_algo, dataset, parsed_lineage, tf_filter=tf_filter, verbose=False, **kwargs)
                    loaded_count += 1
                        
            self.console.print(f"✅Scan complete. Loaded [bold]{loaded_count}[/] networks from '{directory_path}'.")

        # --- Case 2: Source is a DataFrame or a single File Path ---
        else:
            if regex:
                self.console.print("[yellow]Warning: 'regex' parameter is only used when 'source' is a directory.[/]")
            self._add_single_network(source, algo, dataset, lineage, tf_filter=tf_filter, verbose=True, **kwargs)
        
        # After loading, check if the color palette needs to be extended for new algorithms
        algos = set(net["meta"]["Algorithm"] for net in self.networks)
        if len(algos) > len(self.algo_palette):
            self.algo_palette = self._get_algo_palette(list(algos))

        if describe:
            self.describe_networks()

    def _add_single_network(
        self,
        network_data: Union[pd.DataFrame, str],
        algo: str,
        dataset: str,
        lineage: str,
        tf_filter: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Internal method to process and add a single network from a DataFrame or file path.

        Parameters:
        -----------
        network_data (Union[pd.DataFrame, str]): The network data itself, either as a DataFrame or a path to a CSV.
        algo (str): The algorithm name.
        dataset (str): The dataset name.
        lineage (str): The lineage name.
        verbose (bool): If True, prints a confirmation message upon successful addition.
        **kwargs: Additional metadata.

        Return:
        -------
        None
        """
        # Validate that essential metadata is provided
        if not all([algo, dataset, lineage]):
            self.console.print("[red]Error: 'algo', 'dataset', and 'lineage' must be provided for a single network.[/]")
            return
            
        try:
            # Load from file path if a string is provided
            if isinstance(network_data, str):
                network_df = pd.read_csv(network_data)
            # Use the DataFrame directly if provided
            elif isinstance(network_data, pd.DataFrame):
                network_df = network_data.copy()
            else:
                raise TypeError("Input must be a pandas DataFrame or a file path string.")

            if network_df.shape[1] < 3:
                raise ValueError("Network data must have at least 3 columns: TF, Target, and Score.")

            # Standardize column names for consistency
            network_df.columns = ["TF", "Target", "Score"]

            if tf_filter is not None:
                network_df = network_df[network_df["TF"].isin(tf_filter)]

            # Combine provided metadata and any extra kwargs
            meta = {"Algorithm": algo, "Dataset": dataset, "Lineage": lineage, **kwargs}
            # Append the structured network data to the main list
            self.networks.append({"meta": meta, "data": network_df})
            if verbose:
                self.console.print(f"[green]Added network:[/] Algorithm='{algo}', Dataset='{dataset}', Lineage='{lineage}'")

            self.union_tf_list = list(set().union(*(net["data"]["TF"].unique() for net in self.networks)))
            self.union_tg_list = list(set().union(*(net["data"]["Target"].unique() for net in self.networks)))


        except Exception as e:
            self.console.print(f"[red]Error adding network for {algo}/{lineage}: {e}[/]")


    def load_groundtruths(
        self,
        source: Union[pd.DataFrame, str, Dict],
        dataset: str = None,
        lineage: str = None,
        describe: bool = False,
    ):
        """
        Adds ground truth networks from various sources.

        This method can load ground truths from a single file/DataFrame or from a predefined
        dictionary structure mapping (dataset, lineage) tuples to file paths, which is useful
        for loading a standard benchmark set.

        Parameters:
        -----------
        source (Union[pd.DataFrame, str, Dict]): The data source.
            - A dictionary mapping (dataset, lineage) to file paths.
            - A pandas DataFrame for a single ground truth.
            - A file path (str) for a single ground truth.
        dataset (str, optional): The name of the associated dataset. Required for single sources.
        lineage (str, optional): The name of the associated cell lineage. Required for single sources.
        describe (bool): If True, prints the ground truth description table after loading.

        Return:
        -------
        None
        """
        # --- Case 1: Source is a Dictionary (e.g., from constants file) ---
        if isinstance(source, dict):
            loaded_count = 0
            for (ds, lin), path in track(
                source.items(), description="⚙️[bold cyan]Loading Ground Truths..."
            ):
                self._add_single_groundtruth(path, ds, lin, verbose=False)
                # Verify it was added successfully
                if (ds, lin) in self.groundtruths:
                    loaded_count += 1
            self.console.print(f"✅Load complete. Successfully added [bold]{loaded_count}[/] ground truth networks.")

        # --- Case 2: Source is a DataFrame or a single File Path ---
        else:
            self._add_single_groundtruth(source, dataset, lineage, verbose=True)

        if describe:
            self.describe_groundtruths()

    def _add_single_groundtruth(
        self,
        gt_data: Union[pd.DataFrame, str],
        dataset: str,
        lineage: str,
        verbose: bool = True,
    ):
        """
        Internal method to process and add a single ground truth network.

        Parameters:
        -----------
        gt_data (Union[pd.DataFrame, str]): The ground truth data (DataFrame or file path).
        dataset (str): The dataset name.
        lineage (str): The lineage name.
        verbose (bool): If True, prints a confirmation message.

        Return:
        -------
        None
        """
        if not all([dataset, lineage]):
            self.console.print("[red]Error: 'dataset' and 'lineage' must be provided for a single ground truth.[/]")
            return

        try:
            if isinstance(gt_data, str):
                gt_df = pd.read_csv(gt_data)
            elif isinstance(gt_data, pd.DataFrame):
                gt_df = gt_data.copy()
            else:
                raise TypeError("Input must be a pandas DataFrame or a file path string.")

            if gt_df.shape[1] < 2:
                raise ValueError("Ground truth data must have at least 2 columns: TF and Target.")
            
            # Ensure only the first two columns are used, ignoring any extra info
            if gt_df.shape[1] > 2:
                gt_df = gt_df.iloc[:, :2]

            gt_df.columns = ["TF", "Target"]
            # Use a tuple of (dataset, lineage) as the unique key
            key = (dataset, lineage)
            self.groundtruths[key] = gt_df
            if verbose:
                self.console.print(
                    f"[blue]Added ground truth for:[/] dataset='{dataset}', lineage='{lineage}'"
                )
        except Exception as e:
            self.console.print(f"[red]Error adding ground truth for {dataset}/{lineage}: {e}[/]")


    def get_network_info(self, network_df: pd.DataFrame) -> Dict:
        """
        Retrieves basic statistics for a single network DataFrame.

        Parameters:
        -----------
        network_df (pd.DataFrame): A DataFrame with 'TF' and 'Target' columns.

        Return:
        -------
        Dict: A dictionary containing the number of edges, nodes, TFs, and targets.
        """
        return {
            "Edges": len(network_df),
            "Nodes": len(set(network_df["TF"]) | set(network_df["Target"])),
            "TFs": network_df["TF"].nunique(),
            "Targets": network_df["Target"].nunique(),
        }

    def describe_networks(self, query: str = None, regex: Dict[str, str] = None, sort_by: str = None, ascending: bool = True):
        """
        Generates, stores, and prints a rich table summarizing filtered networks.

        This function provides a quick overview of the loaded networks, allowing for filtering
        and sorting to easily inspect their properties.

        Parameters:
        -----------
        query (str, optional): A pandas-style query string to filter networks based on metadata.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering.
        sort_by (str, optional): Column name to sort the table by.
        ascending (bool): Direction of sorting.

        Return:
        -------
        None
        """
        networks_to_describe = self.filter_networks(query=query, regex=regex)
        if not networks_to_describe:
            self.console.print("[yellow]No networks match the specified filters.[/]")
            return
        
        table_data = []
        for net in networks_to_describe:
            info = self.get_network_info(net["data"])
            meta = net["meta"]
            # Combine metadata and network stats into a single dictionary for the row
            table_data.append({"Algorithm": meta["Algorithm"], "Dataset": meta["Dataset"], "Lineage": meta["Lineage"], **info})

        if sort_by:
            valid_keys = ["Algorithm", "Dataset", "Lineage", "Edges", "Nodes", "TFs", "Targets"]
            if sort_by not in valid_keys:
                self.console.print(f"[red]Error: Invalid sort_by key '{sort_by}'. Valid keys are: {valid_keys}[/]")
                return
            
            # Sort numerically for stats columns and alphabetically otherwise
            is_numeric_sort = sort_by in ["Edges", "Nodes", "TFs", "Targets"]
            table_data.sort(
                key=lambda x: int(x[sort_by]) if is_numeric_sort else x[sort_by],
                reverse=not ascending
            )
            
        # Create and format a rich Table
        table = Table(title="Predicted Network Descriptions")
        for col in ["Algorithm", "Dataset", "Lineage", "Edges", "Nodes", "TFs", "Targets"]:
            table.add_column(
                col,
                style="cyan" if col=="Algorithm" else "magenta" if col=="Dataset" else "green" if col=="Lineage" else None,
                justify="right" if col in ["Edges", "Nodes", "TFs", "Targets"] else "left"
            )
        
        for row in table_data:
            table.add_row(row["Algorithm"], row["Dataset"], row["Lineage"], str(row["Edges"]), str(row["Nodes"]), str(row["TFs"]), str(row["Targets"]))
        
        self.grn_info = table  # Cache the table object
        self.console.print(self.grn_info)

    def describe_groundtruths(self, sort_by: str = "Dataset", ascending: bool = True):
        """
        Generates, stores, and prints a rich table summarizing the loaded ground truth networks.

        Parameters:
        -----------
        sort_by (str, optional): Column name to sort by. Defaults to "Dataset".
        ascending (bool): Direction of sorting. Defaults to True.

        Return:
        -------
        rich.table.Table: The generated table object.
        """
        if not self.groundtruths:
            self.console.print("[yellow]No ground truths have been loaded.[/]")
            return None

        table_data = []
        for (dataset, lineage), gt_df in self.groundtruths.items():
            info = self.get_network_info(gt_df)
            table_data.append({"Dataset": dataset, "Lineage": lineage, **info})

        if sort_by:
            valid_keys = ["Dataset", "Lineage", "Edges", "Nodes", "TFs", "Targets"]
            if sort_by not in valid_keys:
                self.console.print(f"[red]Error: Invalid sort_by key '{sort_by}'. Valid keys are: {valid_keys}[/]")
                return
            
            is_numeric_sort = sort_by in ["Edges", "Nodes", "TFs", "Targets"]
            table_data.sort(
                key=lambda x: int(x[sort_by]) if is_numeric_sort else x[sort_by],
                reverse=not ascending
            )

        # Create and format a rich Table
        table = Table(title="Ground Truth Network Descriptions")
        for col in ["Dataset", "Lineage", "Edges", "Nodes", "TFs", "Targets"]:
            table.add_column(
                col,
                style="magenta" if col=="Dataset" else "green" if col=="Lineage" else None,
                justify="right" if col in ["Edges", "Nodes", "TFs", "Targets"] else "left"
            )

        for row in table_data:
            table.add_row(row["Dataset"], row["Lineage"], str(row["Edges"]), str(row["Nodes"]), str(row["TFs"]), str(row["Targets"]))
            
        self.groundtruth_info = table # Store the table
        self.console.print(self.groundtruth_info)
        return table
    
    def filter_networks(self, query: str = None, regex: Dict[str, str] = None) -> List[Dict]:
        """
        Filters the list of stored networks using a pandas-style query string
        and/or regular expressions on their metadata.

        Parameters:
        -----------
        query (str, optional): A query string to filter networks based on their metadata.
                               Example: "Algorithm == 'scVelo' and Lineage in ['Ery', 'Mono']".
        regex (Dict[str, str], optional): A dictionary for regex-based filtering.
                                          Keys are metadata fields (e.g., 'Algorithm'),
                                          and values are regex patterns.
                                          Example: {'Algorithm': r'^sc.*o$', 'Lineage': r'Ery|Mono'}

        Return:
        -------
        List[Dict]: A list of network dictionaries that match the filters.
        """
        if not self.networks:
            return []

        # Create a temporary DataFrame from the metadata for efficient querying
        meta_df = pd.DataFrame([net['meta'] for net in self.networks])
        
        # --- 1. Apply standard query filter (if provided) ---
        if query:
            try:
                meta_df = meta_df.query(query)
            except Exception as e:
                self.console.print(f"[red]Error executing query '{query}': {e}[/]")
                return []

        # --- 2. Apply regex filter (if provided) ---
        if regex:
            if not isinstance(regex, dict):
                self.console.print("[red]Error: The 'regex' parameter must be a dictionary.[/]")
                return []
            
            try:
                for column, pattern in regex.items():
                    # Ensure the column exists to avoid KeyErrors
                    if column not in meta_df.columns:
                        self.console.print(f"[yellow]Warning: Regex column '{column}' not found in metadata. Skipping.[/]")
                        continue
                    # Apply the regex filter using .str.contains()
                    # na=False ensures that missing metadata values don't cause errors
                    meta_df = meta_df[meta_df[column].str.contains(pattern, regex=True, na=False)]
            except Exception as e:
                self.console.print(f"[red]Error applying regex '{pattern}' on column '{column}': {e}[/]")
                return []

        # --- 3. Return the final filtered network objects ---
        matching_indices = meta_df.index
        return [self.networks[i] for i in matching_indices]
        
    def calculate_accuracy(self, thres_mode: str = "topk", baseline_path: str = None):
        """
        Calculates accuracy metrics and stores detailed results, including PR and ROC curve data.

        This is a core evaluation method. It iterates through each loaded network, finds its
        corresponding ground truth, preprocesses both, and then computes a suite of accuracy
        metrics (AUPR, AUROC, F-scores, etc.). It also captures the raw data points for
        Precision-Recall and ROC curves to enable later plotting.

        Parameters:
        -----------
        thres_mode (str): The thresholding mode for F-score calculation ('max' or 'topk').
                          'max' finds the threshold that maximizes the F-score.
                          'topk' sets the threshold to include the top K edges, where K is the number of edges in the ground truth.
        baseline_path (str, optional): Path to a CSV file with existing benchmark results to merge with.

        Return:
        -------
        pd.DataFrame: A DataFrame containing the combined accuracy results.
        """
        self.accuracy_details = []  # Reset detailed results
        new_results_list = []

        for net in track(self.networks, description="[bold green]Calculating Accuracy..."):
            meta, est_grn = net["meta"], net["data"]
            gt_key = (meta["Dataset"], meta["Lineage"])
            # Skip if no corresponding ground truth is loaded
            if gt_key not in self.groundtruths:
                continue

            gt_grn = self.groundtruths[gt_key]
            
            # --- Preprocessing ---
            # Create copies to avoid modifying original data
            est_grn_proc = est_grn.copy()
            gt_grn_proc = gt_grn.copy()
            # Standardize gene names to uppercase
            est_grn_proc[["TF", "Target"]] = est_grn_proc[["TF", "Target"]].apply(lambda x: x.str.upper())
            gt_grn_proc[["TF", "Target"]] = gt_grn_proc[["TF", "Target"]].apply(lambda x: x.str.upper())
            # Remove edges with non-positive scores
            est_grn_proc = est_grn_proc[est_grn_proc.Score >= 0]

            if np.allclose(est_grn_proc["Score"], 1):
                self.console.print(f"[yellow]Warning: All scores are identical in {meta['Algorithm']}/{meta['Dataset']}/{meta['Lineage']}. AUROC may be undefined.[/]")
            # Filter ground truth to only include genes present in the prediction
            valid_tfs = set(est_grn_proc["TF"])
            valid_targets = set(est_grn_proc["Target"])
            gt_grn_proc = gt_grn_proc[gt_grn_proc["TF"].isin(valid_tfs) & gt_grn_proc["Target"].isin(valid_targets)]
            # Remove self-loops and duplicates from ground truth
            gt_grn_proc = gt_grn_proc[gt_grn_proc["TF"] != gt_grn_proc["Target"]].drop_duplicates(keep="first")

            # --- Metric Calculation ---
            # Call external functions to compute metrics and capture curve data
            aupr, precision, recall, aupr_ratio, _ = compute_AUPR(est_grn_proc, gt_grn_proc)
            auroc, fpr, tpr = compute_AUROC(est_grn_proc, gt_grn_proc)
            
            # Store curve data safely, handling cases where it might be None
            pr_curve_data = (recall, precision) if precision is not None and recall is not None else (np.array([]), np.array([]))
            roc_curve_data = (fpr, tpr) if fpr is not None and tpr is not None else (np.array([]), np.array([]))

            epr, _, _ = compute_EPR(est_grn_proc, gt_grn_proc)

            f1score, _, _, _, thres = compute_Fscore(
                est_grn_proc, gt_grn_proc, beta=1.0, thres_mode=thres_mode
            )
            f01score, _, _, _, _ = compute_Fscore(
                est_grn_proc, gt_grn_proc, beta=0.1, thres_mode=thres_mode
            )
            
            # --- Store detailed results ---
            detail_info = {
                "meta": meta,
                "AUPR": aupr, "AUPR Ratio": aupr_ratio, "AUROC": auroc, "EPR": epr,
                f"F1 Score ({thres_mode})": f1score, f"F0.1 Score ({thres_mode})": f01score,
                "threshold": thres,
                "pr_curve": pr_curve_data,  # Store Precision-Recall curve points
                "roc_curve": roc_curve_data # Store ROC curve points
            }
            self.accuracy_details.append(detail_info)
            # Create a separate list for the summary DataFrame, excluding complex objects
            new_results_list.append({**meta, **{k:v for k,v in detail_info.items() if k not in ['meta', 'pr_curve', 'roc_curve']}})

        new_results_df = pd.DataFrame(new_results_list)

        # --- Baseline file handling ---
        if baseline_path:
            try:
                self.console.print(f"\n[cyan]Loading baseline results from:[/] {baseline_path}")
                baseline_df = pd.read_csv(baseline_path)
                # Combine new results with the baseline
                final_df = pd.concat([baseline_df, new_results_df], ignore_index=True)
                # Remove duplicates, keeping the most recent results for a given combination
                final_df.drop_duplicates(subset=["Algorithm", "Dataset", "Lineage"], keep="last", inplace=True)
                self.console.print("Successfully merged new results with the baseline.")
            except FileNotFoundError:
                self.console.print(f"[yellow]Warning: Baseline file not found at '{baseline_path}'. Returning only new results.[/]")
                final_df = new_results_df
            except Exception as e:
                self.console.print(f"[red]Error reading or processing baseline file: {e}. Returning only new results.[/]")
                final_df = new_results_df
        else:
            final_df = new_results_df

        self.accuracy_results = final_df
        return self.accuracy_results

    def filter_accuracy_results(self, query: str = None, regex: Dict[str, str] = None) -> pd.DataFrame:
        """
        Helper function to filter the `self.accuracy_results` DataFrame.

        Parameters:
        -----------
        query (str, optional): A pandas-style query string.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering on columns.

        Return:
        -------
        pd.DataFrame: The filtered DataFrame. Returns an empty DataFrame on error or if no results exist.
        """
        if self.accuracy_results is None:
            self.console.print("[yellow]No accuracy results found. Please run `calculate_accuracy()` first.[/]")
            return pd.DataFrame()
            
        data = self.accuracy_results.copy()
        
        # --- 1. Apply standard query filter ---
        if query:
            try:
                data = data.query(query)
            except Exception as e:
                self.console.print(f"[red]Error executing query '{query}': {e}[/]")
                return pd.DataFrame()

        # --- 2. Apply regex filter ---
        if regex:
            if not isinstance(regex, dict):
                self.console.print("[red]Error: The 'regex' parameter must be a dictionary.[/]")
                return pd.DataFrame()
            
            try:
                for column, pattern in regex.items():
                    if column not in data.columns:
                        self.console.print(f"[yellow]Warning: Regex column '{column}' not found in results. Skipping.[/]")
                        continue
                    data = data[data[column].astype(str).str.contains(pattern, regex=True, na=False)]
            except Exception as e:
                self.console.print(f"[red]Error applying regex '{pattern}' on column '{column}': {e}[/]")
                return pd.DataFrame()
        
        return data

    def calculate_stability(self, query: str = None, regex: Dict[str, str] = None, group_by: str = "Dataset"):
        """
        Calculates pairwise stability metrics (Jaccard, Cosine) for groups of networks.

        This method is used to assess how consistent an algorithm's predictions are across
        different conditions (e.g., across different lineages within the same dataset). It groups
        networks by algorithm and then computes similarity scores for all pairs within each group.

        Parameters:
        -----------
        query (str, optional): A query string to filter which networks to include in the analysis.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering.
        group_by (str): The metadata field used to label the items being compared (e.g., 'Lineage').

        Return:
        -------
        Dict: A dictionary containing the stability results, with keys for each algorithm.
        """
        networks_to_compare = self.filter_networks(query=query, regex=regex)
        if len(networks_to_compare) < 2:
            self.console.print("[yellow]Warning: At least two networks are required to calculate stability.[/]")
            return

        # Group networks by algorithm name first
        grouped_by_algo = {}
        for net in networks_to_compare:
            algo = net["meta"]["Algorithm"]
            if algo not in grouped_by_algo:
                grouped_by_algo[algo] = []
            grouped_by_algo[algo].append(net)
        
        # Calculate stability within each algorithm's set of networks
        for algo, nets in track(grouped_by_algo.items(), description="[bold purple]Calculating Stability..."):
            if len(nets) < 2:
                continue

            # Prepare a dictionary where keys are the 'group_by' labels and values are sets of edges
            data_dict = {
                net["meta"][group_by]: set(
                    net["data"]["TF"] + "|" + net["data"]["Target"]
                )
                for net in nets
            }
            
            # --- Jaccard Similarity ---
            jac_matrix, ax_jac = jaccard_similarity(data_dict, annot=True, title=f"Jaccard Similarity - {algo}")
            plt.show()

            # --- Cosine Similarity ---
            cos_matrix, ax_cos = cosine_similarity(data_dict, annot=True, title=f"Cosine Similarity - {algo}")
            plt.show()

            # Store results as DataFrames for easy access
            self.stability_results[algo] = {
                "jaccard": pd.DataFrame(jac_matrix, index=data_dict.keys(), columns=data_dict.keys()),
                "cosine": pd.DataFrame(cos_matrix, index=data_dict.keys(), columns=data_dict.keys())
            }

        self.console.print("[bold green]Stability calculation complete.[/]")


    def _calculate_centrality(self, G_nx: nx.DiGraph, centrality_func: Callable, nodes: list) -> list:
        """
        Calculate a specific centrality measure for all nodes in a directed graph.

        Parameters
        ----------
            G_nx (networkx.DiGraph): Directed graph.
            centrality_func (Callable): Function to calculate the centrality measure.
            nodes (list): List of nodes to calculate the centrality for.

        Returns
        -------
            list: List of centrality values for the nodes.
        """
        centrality_values = centrality_func(G_nx)
        # Use .get() for robustness, returning 0 if a node isn't in the result
        return [centrality_values.get(node, 0) for node in nodes]
    
    def _network_score(self, G_nx: nx.DiGraph) -> pd.DataFrame:
        """
        Calculate a fixed set of centrality measures for all nodes in a directed graph.

        Parameters
        ----------
            G_nx (networkx.DiGraph): Directed graph.

        Returns
        -------
            pd.DataFrame: DataFrame containing centrality measures for all nodes.
        """
        selected_nodes = list(G_nx.nodes())

        # Define a dictionary mapping centrality measures to their calculation functions
        # centrality_measures: Dict[str, Callable] = {
        #     'degree_centrality': nx.degree_centrality,
        #     'in_degree_centrality': nx.in_degree_centrality,
        #     'out_degree_centrality': nx.out_degree_centrality,
        #     'betweenness_centrality': nx.betweenness_centrality,
        #     'closeness_centrality': nx.closeness_centrality,
        #     'pagerank': nx.pagerank,
        # }

        centrality_measures: Dict[str, Callable] = {
            'degree_centrality': nx.degree_centrality,
        }

        # Create an empty DataFrame to store the results
        result_df = pd.DataFrame(index=selected_nodes)

        # Calculate and assign each centrality measure
        for measure_name, measure_func in centrality_measures.items():
            result_df[measure_name] = self._calculate_centrality(G_nx, measure_func, selected_nodes)

        return result_df
    
    def calculate_network_score(self):
        """
        Calculates various centrality measures and scores for each loaded network.

        For each network in `self.networks`, this method constructs a directed graph
        and computes several node-level metrics:
        - Degree Centrality (in, out, and total)
        - Betweenness Centrality
        - Closeness Centrality
        - PageRank
        - Number of Targets (Out-Degree)

        The results are stored as a DataFrame in a new 'score' key within each
        network's dictionary (e.g., `self.networks[i]['score']`).
        """
        if not self.networks:
            self.console.print("[yellow] No networks loaded. Please add networks before calculating scores.[/]")
            return

        for net in track(self.networks, description="[bold blue] Calculating Network Scores..."):
            try:

                network_df = net["data"]

                G = nx.from_pandas_edgelist(
                    network_df,
                    source="TF",
                    target="Target",
                    edge_attr="Score",
                    create_using=nx.DiGraph(),
                )

                if G.number_of_nodes() > 0:
                    score_df = self._network_score(G)

                    out_degree_dict = dict(G.out_degree())
                    n_targets = pd.Series(out_degree_dict, name="n_targets").reindex(score_df.index, fill_value=0)

                    final_scores = pd.concat([score_df, n_targets], axis=1)

                    net["score"] = final_scores
                else:
                    net["score"] = pd.DataFrame()

            except Exception as e:
                meta = net.get("meta", {})
                algo = meta.get("Algorithm", "N/A")
                lineage = meta.get("Lineage", "N/A")
                self.console.print(f"[red] {algo}/{lineage}  scoring failed: {e}[/]")

        self.console.print("✅ Network scoring complete.")

    def calculate_tf_recovery(
        self,
        tf_regulators_dict: Dict[str, List[str]],
        ranking_metric: str,
        plot: bool = True,
        rank_threshold: int = 2000,
        filter_tfs_in_networks: bool = False, # NEW: Added option to filter the dictionary
    ) -> pd.DataFrame:
        """
        Evaluates the recovery of master transcription factors based on a ranking metric,
        now differentiating between datasets.

        This method ranks TFs using a specified metric and compares this ranking against
        a ground truth list of master regulators for each dataset-lineage combination.
        It calculates the Area Under the Curve (AUC) for the recovery plot, including
        "Optimal" and "Random" baselines, and can visualize the results.
        A `score` DataFrame must be present for each network.
        """
        from copy import deepcopy
        # --- 1. Pre-computation checks ---
        if not hasattr(self, 'networks') or not self.networks or "score" not in self.networks[0] or self.networks[0]["score"].empty:
            print("[red]Error:[/] Network scores not found. Please run `calculate_network_score()` first.")
            return None
        if ranking_metric not in self.networks[0]["score"].columns:
            print(f"[red]Error:[/] Ranking metric '{ranking_metric}' not found in score DataFrames.")
            return None

        # --- NEW: Optional filtering of the ground truth dictionary ---
        # A deepcopy is used to avoid modifying the original dictionary outside this function's scope
        active_tf_regulators_dict = deepcopy(tf_regulators_dict)
        
        if filter_tfs_in_networks:
            print("🔬 Filtering ground truth TFs to include only those present in the networks' TF universe...")

            # Step 1: Get the union of all TFs from all networks
            all_network_tfs = set()
            for net in self.networks:
                if "score" in net and not net["score"].empty:
                    all_network_tfs.update(net["score"].index.str.upper())
            
            if not all_network_tfs:
                print("[yellow]Warning:[/] Could not find any TFs in the network scores to build a filter set.")
                return None

            # Step 2: Filter the ground truth dictionary
            original_counts = {k: len(v) for k, v in active_tf_regulators_dict.items()}
            
            for lineage, gt_tfs in active_tf_regulators_dict.items():
                # Filter the list, ensuring case-insensitivity
                filtered_list = [tf for tf in gt_tfs if tf.upper() in all_network_tfs]
                active_tf_regulators_dict[lineage] = filtered_list

            print("📊 Filtering complete. TF counts per lineage (Original -> Filtered):")
            for lineage, orig_count in original_counts.items():
                filtered_count = len(active_tf_regulators_dict[lineage])
                print(f"  - {lineage}: {orig_count} -> {filtered_count}")

        # --- 2. Collect ranks from algorithm results ---
        all_ranks_data = []
        tfs_per_group = {}
        
        for net in self.networks:
            meta, score_df = net["meta"], net["score"]
            dataset = meta.get("Dataset")
            lineage = meta.get("Lineage")

            # Use the (potentially filtered) dictionary for the check
            if not all([dataset, lineage]) or lineage not in active_tf_regulators_dict:
                continue

            group_key = (dataset, lineage)

            if group_key not in tfs_per_group:
                tfs_per_group[group_key] = set()
            tfs_per_group[group_key].update(score_df.index.str.upper())

            # Use the (potentially filtered) dictionary to get ground truth TFs
            gt_tfs = {tf.upper() for tf in active_tf_regulators_dict[lineage]}
            score_df.index = score_df.index.str.upper()
            ranked_genes = score_df[ranking_metric].sort_values(ascending=False).index.tolist()

            for tf in gt_tfs:
                try:
                    rank = ranked_genes.index(tf) + 1
                except ValueError:
                    rank = np.nan
                all_ranks_data.append({
                    "Dataset": dataset, "Algorithm": meta["Algorithm"], "Lineage": lineage,
                    "TF": tf, "Rank": rank,
                })

        # --- 3. Add baseline ranks (Optimal and Random) ---
        np.random.seed(0)
        for (dataset, lineage), all_tfs in tfs_per_group.items():
            # Use the (potentially filtered) dictionary here as well
            gt_tfs = {tf.upper() for tf in active_tf_regulators_dict.get(lineage, [])}
            gt_tfs_list = sorted(list(gt_tfs))

            # Optimal baseline
            for i, tf in enumerate(gt_tfs_list):
                all_ranks_data.append({
                    "Dataset": dataset, "Algorithm": "Optimal", "Lineage": lineage,
                    "TF": tf, "Rank": i + 1,
                })

            # Random baseline
            shuffled_tfs = list(all_tfs)
            np.random.shuffle(shuffled_tfs)
            for tf in gt_tfs:
                try:
                    rank = shuffled_tfs.index(tf) + 1
                except ValueError:
                    rank = np.nan
                all_ranks_data.append({
                    "Dataset": dataset, "Algorithm": "Random", "Lineage": lineage,
                    "TF": tf, "Rank": rank,
                })
        
        if not all_ranks_data:
            print("[yellow]Warning:[/] No matching ground truth TFs found for the loaded networks.")
            return None

        ranks_df = pd.DataFrame(all_ranks_data)

        # --- 4. Calculate Rank CDF and Normalized Recovery ---
        thresholds = np.arange(1, rank_threshold + 1)
        cdf_data_list = []
        for (dataset, algo, lineage), group_df in ranks_df.groupby(["Dataset", "Algorithm", "Lineage"]):
            ranks = group_df["Rank"].dropna().values
            total_gt_tfs = len(active_tf_regulators_dict.get(lineage, []))
            cdf_counts = (ranks.reshape(-1, 1) <= thresholds.reshape(1, -1)).sum(axis=0)
            
            if total_gt_tfs > 0:
                normalized_rank = cdf_counts / total_gt_tfs
            else:
                normalized_rank = np.zeros_like(cdf_counts, dtype=float)

            cdf_data_list.append(pd.DataFrame({
                "Rank Threshold": thresholds,
                "Rank CDF": cdf_counts,
                "Normalized Rank": normalized_rank,
                "Dataset": dataset, "Algorithm": algo, "Lineage": lineage,
            }))

        cdf_df = pd.concat(cdf_data_list, ignore_index=True)
        self.rank_cdf = cdf_df

        # --- 5. Calculate AUC and Normalized AUC ---
        auc_results = []
        for (dataset, algo, lineage), group_df in cdf_df.groupby(["Dataset", "Algorithm", "Lineage"]):
            auc = group_df["Rank CDF"].sum()
            n_vars = len(active_tf_regulators_dict[lineage]) # Use filtered dict for normalization
            optimal_auc = (n_vars * (n_vars + 1) / 2) + (rank_threshold - n_vars) * n_vars
            normalized_auc = auc / optimal_auc if optimal_auc > 0 else 0
            auc_results.append({
                "Dataset": dataset, "Algorithm": algo, "Lineage": lineage,
                "AUC": auc, "Normalized AUC": normalized_auc,
            })
        
        auc_df = pd.DataFrame(auc_results).sort_values(by=["Dataset", "Lineage", "Algorithm"]).reset_index(drop=True)

        # --- 6. Plotting (optional) ---
        if plot:
            # Assuming _plot_tf_recovery is defined elsewhere in the class
            self.plot_tf_recovery(cdf_df, auc_df) 

        self.rank_auc = auc_df
    

    def plot_tf_recovery(self, cdf_df: pd.DataFrame, auc_df: pd.DataFrame, y_metric: str = "Normalized AUC", save: str = None):
        """
        Helper function to plot master regulator recovery curves, creating a
        separate figure for each dataset.
        """
        if cdf_df.empty or "Dataset" not in cdf_df.columns:
            print("[yellow]Warning:[/] Cannot plot. CDF data is empty or missing 'Dataset' column.")
            return
        
        if auc_df.empty or "Dataset" not in auc_df.columns:
            print("[yellow]Warning:[/] Cannot plot. AUC data is empty or missing 'Dataset' column.")
            return

        all_algos = sorted(cdf_df["Algorithm"].unique())
        real_algos = [a for a in all_algos if a not in ["Optimal", "Random"]]
        # Sort Algorithms by mean y_metric for consistent legend/ordering
        auc_data = auc_df[~auc_df["Algorithm"].isin(["Optimal", "Random"])]
        # Sort Algorithms by mean y_metric for consistent legend/order
        algo_order = auc_data.groupby("Algorithm")[y_metric].mean().sort_values(ascending=False).index
        
        # This is a placeholder for your actual palette function
        # palette = {algo: color for algo, color in zip(real_algos, sns.color_palette("deep", len(real_algos)))}
        palette = self._get_algo_palette(real_algos)
        palette["Optimal"] = "#000000"
        palette["Random"] = "#95a5a6"

        dashes = {algo: "" for algo in real_algos}
        dashes["Optimal"] = (4, 2)
        dashes["Random"] = (1, 1)

        for dataset_name, cdf_dataset_df in cdf_df.groupby("Dataset"):
            lineages = sorted(cdf_dataset_df["Lineage"].unique(), reverse=True)
            if not lineages:
                continue
            
            sns.set_style("whitegrid")
            # fig, axes = plt.subplots(
            #     1, len(lineages), 
            #     figsize=(5 * len(lineages), 4.5), 
            #     sharey=True, # Can set to True now that y-axis is normalized
            #     squeeze=False
            # )

            fig, axes = plt.subplots(
                len(lineages), 2,  
                figsize=(9, 3 * len(lineages)), 
                sharey=False, # Can set to True now that y-axis is normalized
                squeeze=False
            )
            

            for i, lineage in enumerate(lineages):
                lineage_data = cdf_dataset_df[cdf_dataset_df["Lineage"] == lineage]
                ax_line = axes[i, 0]
                ax_bar = axes[i, 1]

                sns.lineplot(
                    data=lineage_data,
                    x="Rank Threshold",
                    y="Normalized Rank", 
                    hue="Algorithm",
                    style="Algorithm",
                    dashes=dashes,
                    palette=palette,
                    hue_order=all_algos,
                    ax=ax_line,
                    legend=(i == len(lineages) - 1) # Put legend on the last subplot
                )
                # ax_line.set_title(f"{lineage}", fontsize=16)
                # MODIFIED: Update the y-axis label
                ax_line.set_ylabel(f"{lineage} \n Normalized Rank", fontsize=14)
                ax_line.set_xlabel("Rank Threshold" if i == len(lineages)-1  else "", fontsize=14)
                ax_line.set_ylim(-0.05, 1.05) # Set y-axis limits from 0 to 1

                auc_dataset_lineage = auc_data[
                    (auc_data["Dataset"] == dataset_name) & (auc_data["Lineage"] == lineage)
                ]

                sns.barplot(
                    data=auc_dataset_lineage,
                    y="Algorithm",
                    x="Normalized AUC", # MODIFIED: Use the new normalized column
                    orient="h",
                    order=algo_order,
                    palette=palette,
                    ax=ax_bar,
                )

                # ax_bar.set_title(f"{lineage} AUC", fontsize=16)
                # ax_bar.set_ylabel("Normalized AUC" if i == 0 else "", fontsize=14)
                # ax_bar.set_yticklabels(ax_bar.get_yticklabels(), fontsize=12)
                ax_bar.tick_params(axis='y', labelsize=12)
                ax_bar.set_xlabel("AUROC" if i == len(lineages)-1 else "", fontsize=14)
                ax_bar.set_ylabel("", fontsize=14)


            fig.suptitle(f"Driver TFs (Dataset: {dataset_name})", fontsize=16, y=1.00)

            # Move legend outside the plot
            if lineages and axes[-1, 0].get_legend() is not None:
                handles, labels = axes[-1, 0].get_legend_handles_labels()
                axes[-1, 0].get_legend().remove()

                # --- KEY CHANGE STARTS HERE ---

                # 1. Create a dictionary mapping each label (algorithm name) to its handle
                label_handle_map = dict(zip(labels, handles))

                # 2. Define the desired order for the legend. 
                #    `all_algos` is already sorted and includes "Optimal" and "Random".
                desired_order = algo_order.tolist() + ["Optimal", "Random"]

                # 3. Create new lists of handles and labels based on the desired order.
                #    We use .get() to avoid errors if a label from the plot isn't in our desired_order list.
                ordered_handles = [label_handle_map.get(label) for label in desired_order if label in label_handle_map]
                ordered_labels = [label for label in desired_order if label in label_handle_map]
                
                # --- KEY CHANGE ENDS HERE ---

                fig.legend(handles=ordered_handles, labels=ordered_labels, 
                            loc='lower center', bbox_to_anchor=(0.5, 0.05),
                            ncol=len(ordered_labels), fontsize=10,  frameon=True, framealpha=0.9, borderpad=0.5)


            fig.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust rect to make space for legend
            plt.show()

            if save:
                if save.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf', '.svg')):
                    filename = save
                else:
                    filename = f"{save}.pdf"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"💾 Saved TF recovery plot for dataset '{dataset_name}' to: {filename}")


    def plot_recovery_curves(
        self,
        cdf_df: pd.DataFrame,
        ncols: int = 4,
        save: Optional[str] = None
    ):
        """
        Plots master regulator recovery curves in a grid layout.
        
        Corrected version that robustly handles legend creation.
        """
        if cdf_df.empty or "Dataset" not in cdf_df.columns:
            self.print("[yellow]Warning:[/] Cannot plot. CDF data is empty or missing 'Dataset' column.")
            return

        # --- 1. Setup Palettes and Styles ---
        all_algos = sorted(cdf_df["Algorithm"].unique())
        real_algos = [a for a in all_algos if a not in ["Optimal", "Random"]]
        
        palette = self._get_algo_palette(real_algos)
        palette["Optimal"] = "#000000"
        palette["Random"] = "#95a5a6"

        dashes = {algo: "" for algo in real_algos}
        dashes["Optimal"] = (4, 2)
        dashes["Random"] = (1, 1)

        # --- 2. Iterate Through Each Dataset to Create a Figure ---
        for dataset_name, cdf_dataset_df in cdf_df.groupby("Dataset"):
            lineages = sorted(cdf_dataset_df["Lineage"].unique())
            if not lineages:
                continue
            
            n_lineages = len(lineages)

            # --- 3. Calculate Grid Layout ---
            nrows = (n_lineages + ncols - 1) // ncols

            sns.set_style("whitegrid")
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, squeeze=False
            )
            axes = axes.flatten()

            # --- 4. Plot Each Lineage on a Subplot ---
            for i, lineage in enumerate(lineages):
                ax = axes[i]
                lineage_data = cdf_dataset_df[cdf_dataset_df["Lineage"] == lineage]

                # IMPORTANT: Enable the legend on each plot initially so we can collect its items
                sns.lineplot(
                    data=lineage_data, x="Rank Threshold", y="Normalized Rank",
                    hue="Algorithm", style="Algorithm", dashes=dashes, palette=palette,
                    hue_order=all_algos, ax=ax,
                    
                )
                
                ax.set_title(f"{lineage}", fontsize=14)
                ax.set_ylim(-0.05, 1.05)
                
                if i % ncols == 0:
                    ax.set_ylabel("Normalized Rank", fontsize=12)
                else:
                    ax.set_ylabel("")
                
                if i >= n_lineages - ncols:
                    ax.set_xlabel("Rank Threshold", fontsize=12)
                else:
                    ax.set_xlabel("")

            # --- 5. Hide Unused Subplots ---
            for i in range(n_lineages, len(axes)):
                axes[i].set_visible(False)
            
            # --- 6. Create and Place a Single Shared Legend (ROBUST METHOD) ---
            
            # (6a) Collect unique legend handles and labels from ALL subplots
            master_label_handle_map = {}
            for i in range(n_lineages):
                ax = axes[i]
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in master_label_handle_map:
                        master_label_handle_map[label] = handle
                # (6b) Remove the individual subplot legends now that we have their info
                if ax.get_legend():
                    ax.get_legend().remove()

            # (6c) Create the final ordered legend lists
            desired_order = all_algos
            ordered_handles = [master_label_handle_map.get(label) for label in desired_order if label in master_label_handle_map]
            ordered_labels = [label for label in desired_order if label in master_label_handle_map]
            
            # (6d) Add the legend to the figure only if items were found
            if ordered_labels:
                fig.legend(
                    handles=ordered_handles, labels=ordered_labels,
                    loc='lower center', bbox_to_anchor=(0.5, 0),
                    ncol=len(ordered_labels), fontsize=11, frameon=False
                )

            fig.suptitle(f"Driver TF Recovery (Dataset: {dataset_name})", fontsize=16, y=1.0)
            fig.tight_layout(rect=[0, 0.05, 1, 0.97])
            plt.show()
            
            # --- 7. Save the Figure if Requested ---
            if save:
                base, ext = (save, 'pdf')
                if '.' in save:
                    parts = save.split('.')
                    base = '.'.join(parts[:-1]); ext = parts[-1]
                
                filename = f"{base}_{dataset_name}.{ext}"
                fig.savefig(filename, dpi=300, bbox_inches='tight')

    def plot_scatter(self, x_metric: str, y_metric: str, query: str = None, regex: Dict[str, str] = None, save: str = None, **kwargs):
        """
        Plots a scatter plot with error bars for two specified accuracy metrics.

        This method aggregates results by algorithm to show the mean performance and variability
        (standard deviation) for two metrics, which is useful for visualizing trade-offs.

        Parameters:
        -----------
        x_metric (str): The name of the metric for the x-axis.
        y_metric (str): The name of the metric for the y-axis.
        query (str, optional): A query string to filter the accuracy results before plotting.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering of results.
        **kwargs: Additional keyword arguments passed to the underlying plotting function.

        Return:
        -------
        None
        """
        # Filter the data based on the query and/or regex
        data = self.filter_accuracy_results(query=query, regex=regex)
        
        if data.empty:
            self.console.print("[yellow]No data matches the filters. Cannot generate plot.[/]")
            return
            
        # Use the external plotting function for a standardized look
        self.console.print(f"[bold cyan]Generating scatter plot for '{x_metric}' vs '{y_metric}'...[/]")
        try:
            # Dynamically create a palette for the algorithms present in the filtered data
            algos = data["Algorithm"].unique().tolist()
            palette = self._get_algo_palette(algos)
            print(palette)
            ax = plot_scatter_with_error_bars(
                data=data,
                x_metric=x_metric,
                y_metric=y_metric,
                palette=palette,
                **kwargs
            )
            plt.show()

            if save:
                if save.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf', '.svg')):
                    filename = save
                else:
                    filename = f"{save}.pdf"
                ax.figure.savefig(filename, dpi=300, bbox_inches='tight')
                self.console.print(f"💾 Saved scatter plot to: {filename}")

        except Exception as e:
            self.console.print(f"[red]An error occurred during plotting: {e}[/]")

        

    def plot_violin(self, metric: str, query: str = None, regex: Dict[str, str] = None, **kwargs):
        """
        Generates a violin plot to show the distribution of a specified metric for each algorithm.

        Violin plots are effective for comparing the distribution of performance scores across
        different algorithms, showing both the density and range of results.

        Parameters:
        -----------
        metric (str): The name of the metric to plot from the accuracy results.
        query (str, optional): A query string to filter results before plotting.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering of results.
        **kwargs: Additional keyword arguments passed to the `plot_violin` function.

        Return:
        -------
        None
        """
        data_to_plot = self.filter_accuracy_results(query=query, regex=regex)
        if data_to_plot.empty:
            self.console.print("[yellow]No data matches the filters. Cannot generate violin plot.[/]")
            return
            
        # Get dynamic palette for the relevant algorithms
        algos = data_to_plot["Algorithm"].unique().tolist()
        palette = self._get_algo_palette(algos)
        
        # Create the plot using the external helper function
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (max(6, len(algos) * 0.8), 5)))
        plot_violin(
            data=data_to_plot,
            x="Algorithm",
            y=metric,
            ax=ax,
            palette=palette,
            **kwargs
        )
        ax.set_title(f"Distribution of {metric}")
        plt.show()

    def plot_acc(self, x_metric: str, y_metric: str, query: str = None, regex: Dict[str, str] = None, save: str = None, **kwargs):
        """
        Generates a combined accuracy plot with scatter and violin subplots.
        This comprehensive plot visualizes the relationship between two accuracy metrics
        while also showing the distribution of each metric across algorithms.

        Parameters:
        -----------
        x_metric (str): The name of the metric for the x-axis in the scatter plot.
        y_metric (str): The name of the metric for the y-axis in the scatter plot.
        query (str, optional): A query string to filter results before plotting.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering of results.
        **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Return:
        -------
        None
        """
        data_to_plot = self.filter_accuracy_results(query=query, regex=regex)
        if data_to_plot.empty:
            self.console.print("[yellow]No data matches the filters. Cannot generate combined plot.[/]")
            return
            
        # Get dynamic palette for the relevant algorithms
        algos = data_to_plot["Algorithm"].unique().tolist()
        palette = self._get_algo_palette(algos)

        # Create a combined figure with scatter and violin subplots
        fig = plt.figure(figsize=kwargs.pop('figsize', (10, 5)), dpi=kwargs.pop('dpi', 150))
        # gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        # ax_scatter = fig.add_subplot(gs[0, :])
        # ax_violin_x = fig.add_subplot(gs[1, 0])
        # ax_violin_y = fig.add_subplot(gs[1, 1])

        ax_scatter = plt.subplot(2,2, (1,3))
        ax_violin_x = plt.subplot(2,2,2)
        ax_violin_y = plt.subplot(2,2,4)

        # Scatter plot with error bars
        ax_scatter = plot_scatter_with_error_bars(
            data=data_to_plot,
            x_metric=x_metric,
            y_metric=y_metric,
            ax=ax_scatter,
            palette=palette,
            spline_linewidth=1,
            **kwargs
        )

        ax_scatter.get_legend().remove()  
        # Violin plot for x_metric
        ax_violin_x = plot_violin(
            data=data_to_plot,
            x="Algorithm",
            y=x_metric,
            ax=ax_violin_x,
            palette=palette,
            spline_linewidth=1,
            xticklabels=[],
            **kwargs
        )
        ax_violin_x.set_xlabel(f"{x_metric}", fontsize=12)

        # Violin plot for y_metric
        ax_violin_y = plot_violin(
            data=data_to_plot,
            x="Algorithm",
            y=y_metric,
            ax=ax_violin_y,
            palette=palette,
            spline_linewidth=1,
            xticklabels=[],
            **kwargs
        )
        ax_violin_y.set_xlabel(f"{y_metric}", fontsize=12)

        handles, labels = ax_scatter.get_legend_handles_labels()
        print(labels)
        n_algo = len(algos)
        fig.legend(handles=handles, labels=labels, 
                loc='lower center', bbox_to_anchor=(0.5, -0.05),
                columnspacing=0.2, handletextpad=0.1,
                ncol=n_algo, fontsize=10,  frameon=True, framealpha=0.9, borderpad=0.5)

        fig.tight_layout()

        if save:
            if save.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf', '.svg')):
                filename = save
            else:
                filename = f"{save}.pdf"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.console.print(f"💾 Saved combined accuracy plot to: {filename}")
        
    def plot_barplot(self, metric: str, query: str = None, regex: Dict[str, str] = None, **kwargs):
        """
        Generates a bar plot for a specified metric, grouped for comparison.

        This plot is useful for comparing algorithm performance on specific tasks,
        typically grouped by lineage and dataset on the x-axis.

        Parameters:
        -----------
        metric (str): The name of the metric to plot.
        query (str, optional): A query string to filter results before plotting.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering of results.
        **kwargs: Additional keyword arguments passed to `sns.barplot`.

        Return:
        -------
        None
        """
        data_to_plot = self.filter_accuracy_results(query=query, regex=regex)
        if data_to_plot.empty:
            self.console.print("[yellow]No data matches the filters. Cannot generate bar plot.[/]")
            return

        # Combine Lineage and Dataset for a unique x-axis category
        data_to_plot["x_combined"] = data_to_plot["Lineage"] + "\n" + data_to_plot["Dataset"]
        
        algos = data_to_plot["Algorithm"].unique().tolist()
        palette = self._get_algo_palette(algos)

        # Create the plot
        figsize = kwargs.pop('figsize', (15, 6))
        fig, ax = plt.subplots(figsize=figsize)
        sns.set_style("ticks")

        sns.barplot(
            data=data_to_plot,
            x="x_combined",
            y=metric,
            hue="Algorithm",
            palette=palette,
            ax=ax,
            **kwargs
        )
        
        ax.set_ylabel(metric)
        ax.set_xlabel(None)
        ax.legend(title=None, loc="best")
        plt.tight_layout()
        plt.show()
    
    def plot_performance_curves(
        self,
        curve_type: str,
        group_by: str,
        query: str = None,
        regex: Dict[str, str] = None,
        figsize_per_group: Tuple[int, int] = (5, 4),
        interpolate: bool = True,
        xlim: List[float] = [0, 1],
        ylim: List[float] = [0, 1],
        ncols: int = 4, # <-- New parameter to control the number of columns
        save: str = None,
        **kwargs
    ):
        """
        Plots Precision-Recall or ROC curves, grouped into subplots by a metadata field.

        This powerful function visualizes the trade-off between precision/recall or TPR/FPR.
        It can average multiple curves for the same algorithm using interpolation to show a
        mean performance curve with a confidence interval.

        Parameters:
        -----------
        curve_type (str): Type of curve to plot ('pr' or 'roc').
        group_by (str): The metadata field to create subplots for (e.g., 'Dataset', 'Lineage').
        query (str, optional): A query string to filter results before plotting.
        figsize_per_group (Tuple[int, int]): Figure size for each subplot.
        interpolate (bool, optional): If True (default), interpolates curves onto a common axis to
                                      produce a smooth mean curve and confidence interval. If False,
                                      raw points are used directly.
        ncols (int, optional): Number of columns for the subplot grid. Defaults to 4.
        save (str, optional): Filepath to save the figure. The format is inferred from the extension.
        **kwargs: Additional keyword arguments passed to `sns.lineplot`.

        Return:
        -------
        None
        """
        if not self.accuracy_details:
            self.console.print("[bold red]Error:[/] Please run `calculate_accuracy()` first.")
            return

        # --- 1. Validate Input and set labels ---
        if curve_type.lower() == 'pr':
            curve_key, x_label, y_label = "pr_curve", "Recall", "Precision"
        elif curve_type.lower() == 'roc':
            curve_key, x_label, y_label = "roc_curve", "False Positive Rate", "True Positive Rate"
        else:
            raise ValueError("`curve_type` must be either 'pr' or 'roc'.")

        # --- 2. Filter the detailed results based on the query ---
        if query:
            meta_df = pd.DataFrame([res['meta'] for res in self.accuracy_details])
            try:
                matching_indices = meta_df.query(query).index
                results_to_plot = [self.accuracy_details[i] for i in matching_indices]
            except Exception as e:
                self.console.print(f"[red]Error executing query '{query}': {e}[/]")
                return

        else:
            results_to_plot = self.accuracy_details
        
        if not results_to_plot:
            self.console.print("[yellow]No data matches the query. Cannot generate plot.[/]")
            return

        # --- 3. Prepare a long-form DataFrame for Seaborn ---
        plot_data_list = []
        if interpolate:
            base_x_interp = np.linspace(0, 1, 101)
            for result in results_to_plot:
                meta = result['meta']
                if curve_key not in result or result[curve_key] is None or len(result[curve_key][0]) == 0:
                    continue
                
                x_vals, y_vals = result[curve_key]
                sort_indices = np.argsort(x_vals)
                x_vals_sorted, y_vals_sorted = x_vals[sort_indices], y_vals[sort_indices]
                
                interp_y_vals = np.interp(base_x_interp, x_vals_sorted, y_vals_sorted)
                temp_df = pd.DataFrame({x_label: base_x_interp, y_label: interp_y_vals})
                
                for key, val in meta.items():
                    temp_df[key] = val
                plot_data_list.append(temp_df)
        else:
            for result in results_to_plot:
                meta = result['meta']
                if curve_key not in result or result[curve_key] is None or len(result[curve_key][0]) == 0:
                    continue
                
                x_vals, y_vals = result[curve_key]
                temp_df = pd.DataFrame({x_label: x_vals, y_label: y_vals})
                
                for key, val in meta.items():
                    temp_df[key] = val
                plot_data_list.append(temp_df)

        if not plot_data_list:
            self.console.print(f"[yellow]No valid curve data found for '{curve_type}' after filtering.[/]")
            return

        plot_df = pd.concat(plot_data_list, ignore_index=True)
        if group_by == "Lineage":
            plot_df["combined"] = plot_df["Lineage"] + " (" + plot_df["Dataset"] + ")"
            group_by = "combined"

        # --- 4. Plotting ---
        groups = sorted(plot_df[group_by].unique())
        n_groups = len(groups)
        algos = plot_df["Algorithm"].unique().tolist()
        palette = self._get_algo_palette(algos)
        
        # --- MODIFICATION START ---
        # Calculate grid dimensions based on ncols
        if n_groups < ncols:
            ncols = n_groups
        nrows = (n_groups + ncols - 1) // ncols # Ceiling division to get number of rows

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_group[0] * ncols, figsize_per_group[1] * nrows),
            sharey=True,
            squeeze=False # Ensure axes is always a 2D array for consistency
        )
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
        # --- MODIFICATION END ---

        for i, group in enumerate(groups):
            ax = axes[i]
            group_df = plot_df[plot_df[group_by] == group]
            
            sns.lineplot(
                data=group_df,
                x=x_label,
                y=y_label,
                hue="Algorithm",
                palette=palette,
                ax=ax,
                ci=95 if interpolate else None,
                estimator='median' if interpolate else None,
                lw=2,
                **kwargs
            )
            ax.set_title(f"{group_by}: {group}")
            if ax.get_legend():
                ax.get_legend().remove()

            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        # --- MODIFICATION START ---
        # Hide any unused subplots in the grid
        for i in range(n_groups, len(axes)):
            axes[i].set_visible(False)
        # --- MODIFICATION END ---

        # Create a single, shared legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        # Order the legend by the order of algorithms in the palette
        ordered_handles_labels = sorted(zip(handles, labels), key=lambda hl: algos.index(hl[1]) if hl[1] in algos else len(algos))
        fig.legend(handles=[hl[0] for hl in ordered_handles_labels],
                   labels=[hl[1] for hl in ordered_handles_labels],
                   loc="lower center", ncol=len(algos), bbox_to_anchor=(0.5, 0),
                   frameon=False)
        
        fig.suptitle(f'{"Precision-Recall" if curve_type == "pr" else "ROC"} Curves', fontsize=16)
        
        # --- MODIFICATION START ---
        # Adjust layout to prevent title and legend from overlapping with plots
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        # --- MODIFICATION END ---

        plt.show()

        if save:
            if not any(save.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf', '.svg']):
                filename = f"{save}.pdf"
            else:
                filename = save
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.console.print(f"💾 Saved performance curves to: {filename}")
        
    def plot_score_distributions(
        self,
        group_by: str,
        query: str = None,
        regex: Dict[str, str] = None,
        ncols: int = 3,
        figsize_per_plot: Tuple[int, int] = (6, 4),
        bins: int = 50,
        kde: bool = True,
        log_scale: bool = False,
        **kwargs
    ):
        """
        Plots the score distributions for multiple networks, grouped into subplots.

        This visualization helps to understand the range and shape of edge scores produced
        by different algorithms, which can reveal biases or different scoring strategies.

        Parameters:
        -----------
        group_by (str): The metadata field to create subplots for (e.g., 'Dataset').
        query (str, optional): A query string to filter which networks to include.
        regex (Dict[str, str], optional): A dictionary for regex-based filtering.
        ncols (int): Number of columns for the subplot grid.
        figsize_per_plot (Tuple[int, int]): Figure size for each individual subplot.
        bins (int): Number of bins for the histogram.
        kde (bool): Whether to include a kernel density estimate (KDE).
        log_scale (bool): If True, sets the x-axis (Score) to a logarithmic scale.
        **kwargs: Additional keyword arguments passed to `sns.histplot`.
        
        Return:
        -------
        None
        """
        if not self.networks:
            self.console.print("[yellow]No networks loaded. Please run `load_grns()` first.[/]")
            return
        if not self.accuracy_details:
            self.console.print("[yellow]Warning: Accuracy not calculated. F1 thresholds will not be shown.[/]")

        networks_to_plot = self.filter_networks(query=query, regex=regex)
        if not networks_to_plot:
            self.console.print("[yellow]No networks match the query.[/]")
            return

        from collections import defaultdict
        grouped_data = defaultdict(list)
        for net in networks_to_plot:
            group_name = net["meta"].get(group_by)
            if group_name:
                grouped_data[group_name].append(net)

        # Setup subplot grid dimensions
        n_plots = len(grouped_data)
        nrows = (n_plots - 1) // ncols + 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            constrained_layout=True
        )
        axes = axes.flatten()
        sns.set_theme(style="whitegrid")

        for i, (group_name, nets_in_group) in enumerate(sorted(grouped_data.items())):
            ax = axes[i]
            for net in nets_in_group:
                algo = net["meta"]["Algorithm"]
                scores = net["data"]["Score"]
                color = self._get_algo_palette([algo])[algo]

                # Plot the distribution
                sns.histplot(
                    scores, bins=bins, kde=kde, ax=ax, label=algo, color=color, alpha=0.6, log_scale=log_scale, **kwargs
                )

            ax.set_title(f"Score Distribution for {group_by}: {group_name}", weight="bold")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.legend(title="Algorithm", loc="best")

        # Hide any unused subplots in the grid
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.show()
    
    def _get_algo_palette(self, algos: List[str]) -> Dict[str, str]:
        """
        Generates a consistent and visually distinct color palette for a list of algorithms.

        It prioritizes predefined colors from `METHOD_PALETTE` and assigns new, non-overlapping
        colors for any algorithms not already defined.

        Parameters:
        -----------
        algos (List[str]): A list of unique algorithm names.

        Return:
        -------
        Dict[str, str]: A dictionary mapping algorithm names to color hex codes.
        """
        from matplotlib.colors import to_hex, to_rgb
        palette = self.algo_palette.copy()

        # Find which algorithms don't have a color assigned yet
        unassigned_algos = [algo for algo in algos if algo not in palette]
        
        if unassigned_algos:
            used_colors_rgb = {to_rgb(color) for color in palette.values()}
            
            # Generate a pool of visually distinct candidate colors using 'husl'
            num_needed = len(unassigned_algos)
            candidate_pool_size = num_needed + len(used_colors_rgb) + 10 # Buffer to ensure we find enough
            candidate_colors = sns.color_palette("husl", n_colors=candidate_pool_size)
            
            # Filter out colors that are already in use
            new_colors = []
            for color in candidate_colors:
                if tuple(color[:3]) not in used_colors_rgb:
                    new_colors.append(color)
                if len(new_colors) == num_needed:
                    break
            
            if len(new_colors) < num_needed:
                # Fallback or error if not enough unique colors can be found
                raise ValueError("Could not generate enough unique colors for the new algorithms.")
                
            # Assign the new colors to the unassigned algorithms
            for i, algo in enumerate(unassigned_algos):
                palette[algo] = to_hex(new_colors[i])
                
        # Return a dictionary containing only the colors for the requested algorithms
        return {algo: palette[algo] for algo in algos}
    
    
    def show_algo_palette(self):
        """Displays the current algorithm color palette as a visual chart."""
        if not self.algo_palette:
            self.console.print("[yellow]No algorithms have been loaded yet.[/]")
            return
        
        from grn_tools._plotting import show_color_dict
        # Use an external helper function to render the color dictionary
        show_color_dict(self.algo_palette)

    def show_grn_tree(self):
        """
        Displays a tree structure of the loaded networks and their corresponding ground truths.

        This provides a quick, hierarchical overview of the evaluation setup, making it easy
        to see which networks have been loaded and which have a matching ground truth available
        for comparison.

        Return:
        -------
        None
        """
        from rich.tree import Tree
        from collections import defaultdict

        if not self.networks:
            self.console.print("[yellow]No GRNs have been loaded. Use 'load_grns()' to add networks.[/]")
            return

        # Group networks by Dataset -> Lineage -> Algorithm for a nested structure
        tree_data = defaultdict(lambda: defaultdict(list))
        for net in self.networks:
            meta = net['meta']
            tree_data[meta['Dataset']][meta['Lineage']].append(meta['Algorithm'])

        # Create the root of the tree
        tree = Tree(
            "🔬 [bold cyan]Loaded Gene Regulatory Networks[/]",
            guide_style="bold bright_blue"
        )

        # Build the tree structure by iterating through the grouped data
        for dataset, lineages in sorted(tree_data.items()):
            dataset_branch = tree.add(f"📚 [magenta]Dataset: {dataset}[/]")
            for lineage, algos in sorted(lineages.items()):
                # Check if a ground truth exists for this specific combination
                gt_key = (dataset, lineage)
                gt_status = "✅ [green]Ground Truth Loaded[/]" if gt_key in self.groundtruths else "❌ [red]No Ground Truth[/]"
                
                lineage_branch = dataset_branch.add(f"🧬 [green]Lineage: {lineage}[/] ({gt_status})")
                
                # Add algorithms as the final leaves of the branch
                for algo in sorted(algos):
                    lineage_branch.add(f"⚙️ [cyan]Algorithm: {algo}[/]")
        
        self.console.print(tree)