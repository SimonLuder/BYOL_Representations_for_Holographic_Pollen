import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Tuple, Union
import scipy.stats as stats
from ssl_poleno.evaluation.report import EvaluationSummary

from ssl_poleno.evaluation import knn


def load_inference_results(filename: str) -> pd.DataFrame:
    data = np.load(filename)

    def make_repr(emb_key, proj_key, files_key):
        d = {
            "emb": data[emb_key].tolist(),
            "rec_path": data[files_key].tolist(),
        }
        if proj_key in data:
            d["proj"] = data[proj_key].tolist()
        return pd.DataFrame(d)

    repr1 = make_repr("emb1", "proj1", "files1")
    repr2 = make_repr("emb2", "proj2", "files2")

    df = pd.concat([repr1, repr2], ignore_index=True)
    labels_name = data.get("dataset")
    return df, labels_name


def calc_cv_accuracy_ci(fold_acc: List[float]):

    K = len(fold_acc)

    mean_acc = fold_acc.mean()
    sd = fold_acc.std(ddof=1)          # sample SD
    se = sd / np.sqrt(K)

    alpha = 0.05
    tcrit = stats.t.ppf(1 - alpha/2, df=K-1)

    ci_low = mean_acc - tcrit * se
    ci_high = mean_acc + tcrit * se

    return mean_acc, (ci_low, ci_high)


def calc_sd_se(fold_acc: List[float]):
    K = len(fold_acc)
    sd = fold_acc.std(ddof=1)     # sample SD across folds
    se = sd / np.sqrt(K)
    return sd, se


def find_files_by_regex(root_dir: Union[str, Path], pattern: str) -> List[Path]:
    root_dir = Path(root_dir)
    regex = re.compile(pattern)

    matches = []
    for path in root_dir.rglob("*"):
        if path.is_file() and regex.fullmatch(path.name):
            matches.append(path)

    return matches


def run_tests(checkpoints, labels, ckpt_root="checkpoints", k_fold = 5, k_neighbors=10):

    # Load ground truth labels
    test_labels = pd.read_csv(labels)

    results = []
    for checkpoint in checkpoints:

        files = find_files_by_regex(
            root_dir=os.path.join(ckpt_root, checkpoint), 
            pattern=r"inference_.*\.npz"
        )

        for filename in files:
            print(f"Calculate respresentations for file: {filename}")
            # filename = f"{ckpt_root}/{checkpoint}/inference/inference_last.npz"
            representations, labels_name = load_inference_results(filename)

            df = pd.merge(representations, test_labels, on="rec_path", how='inner')
            df = df.sort_values(["species", "event_id"]).reset_index(drop=True)

            out = knn.evaluate_embeddings_knn_cv(df, y_col="species", k=k_neighbors, n_splits=k_fold)
            predictions, true_labels, test_indices, accuracies = out

            mean_acc, ci = calc_cv_accuracy_ci(np.array(accuracies))
            acc_se, acc_sd = calc_sd_se(np.array(accuracies))

            version = os.path.basename(os.path.dirname(filename))

            result = {
                "checkpoint": checkpoint,
                "version": version,
                "labels": labels_name,
                "mean_cv_accuracy": mean_acc,
                "cv_accuracy_ci_low_95": ci[0],
                "cv_accuracy_ci_high_95": ci[1],
                "cv_accuracy_sd": acc_sd,
                "cv_accuracy_se": acc_se,
                "k_fold": k_fold,
                "k_neighbours": k_neighbors,
            }
            results.append(result)

    return results

        
if __name__ == "__main__":

    labels = r"Z:\simon_luder\Data_Setup\Pollen_Datasets\data\final\poleno\combined_test_20.csv"
    
    parser = argparse.ArgumentParser(description='Arguments for postprocessing')

    parser.add_argument(
        '--outfile',  
        default="evaluation_summary.csv", 
        type=str,
    )

    parser.add_argument(
        '--ckpts', 
        dest='ckpt_names', 
        nargs='+', 
        default=None, 
        type=str, 
        help='List of checkpoint / model names'
    )
    
    parser.add_argument(
        '--labels', 
        dest='labels', 
        nargs='+', 
        default=None, 
        type=str, 
        help='List of test labels'
    )
    
    args = parser.parse_args()

    summary = EvaluationSummary(args.outfile, overwrite=True)
    
    checkpoint_names = [
        "byol_lit_20260129_235610",
        "byol_lit_20260205_131039",
        "byol_lit_20260202_232449",
        "byol_lit_20260204_105645",
    ]

    baselines = [
        "clip_vision",
        "dinov2_vision",
    ]

    if args.ckpt_names is not None:
        checkpoint_names = args.ckpt_names

    checkpoints = checkpoint_names + baselines

    if args.labels is not None:
        labels = args.labels

    results = run_tests(checkpoints, labels)

    for i, new_eval in enumerate(results):
        summary.add_evaluation(new_eval, checkpoint_key=i)
        summary.save()

    print(summary)
