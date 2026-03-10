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
from ssl_poleno.evaluation.mrr import calc_mrr_pd

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



def split_species_groups(df: pd.DataFrame, col="species_norm", n_groups=5, seed=42):
    species = df[col].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(species)

    return np.array_split(species, n_groups)


def run_tests_species_sizes(
        checkpoints,
        labels,
        embeddings,
        ckpt_root="checkpoints",
        k_fold=5,
        k_neighbors=10,
        train_sizes=[10, 20, 40, 80],
):

    test_labels = pd.read_csv(labels)
    results = []

    for checkpoint in checkpoints:

        # files = find_files_by_regex(
        #     root_dir=os.path.join(ckpt_root, checkpoint),
        #     pattern=r"inference_.*\.npz"
        # )

        filename = os.path.join(ckpt_root, checkpoint, embeddings)

        print(f"Calculate representations for file: {filename}")

        representations, labels_name = load_inference_results(filename)

        df = pd.merge(representations, test_labels, on="rec_path", how="inner")
        df = df.sort_values(["species", "event_id"]).reset_index(drop=True)

        version = os.path.basename(os.path.dirname(filename))

        n_species = df["species"].nunique()
        n_samples = len(df)

        print(f"Dataset contains {n_species} species and {n_samples} samples")

        # # compute event retrieval once
        # event_mrr = calc_mrr_pd(df, emb_col="emb", lbl_col="event_id")

        for train_size in train_sizes:

            print(
                f"Running kNN with {train_size} training samples per class"
            )

            out = knn.evaluate_embeddings_knn_cv(
                df,
                y_col="species",
                k=k_neighbors,
                n_splits=k_fold,
                train_samples_per_class=train_size
            )

            predictions, true_labels, test_indices, accuracies = out

            accuracies = np.array(accuracies)

            mean_acc, ci = calc_cv_accuracy_ci(accuracies)
            acc_sd, acc_se = calc_sd_se(accuracies)

            result = {
                "checkpoint": checkpoint,
                "version": version,
                "labels": labels_name,

                "train_samples_per_class": train_size,

                "species_total": n_species,
                "samples_total": n_samples,

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

    labels = [
        "Z:/simon_luder/Data_Setup/Pollen_Datasets/data/final/poleno/combined_test.csv",
    ]
    
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
    
    parser = argparse.ArgumentParser(description='Arguments for postprocessing')

    parser.add_argument(
        '--outfile',  
        default="evaluation_summary_subsets.csv", 
        type=str,
    )

    parser.add_argument(
        '--ckpts', 
        dest='ckpt_names', 
        nargs='+', 
        default=checkpoint_names, 
        type=str, 
        help='List of checkpoint / model names'
    )
    
    parser.add_argument(
        '--labels', 
        default=labels, 
        nargs='+', 
        type=str, 
        help='List of test labels'
    )

    parser.add_argument(
        '--embeddings', 
        default="inference/knn/inference_basic_test.npz", 
        type=str, 
        help='Embeddings file'
    )
    
    args = parser.parse_args()

    summary = EvaluationSummary(args.outfile, overwrite=True)

    checkpoints = args.ckpt_names + baselines

    eval_idx = 0
    for label_file in args.labels:
        results = run_tests_species_sizes(checkpoints, label_file, args.embeddings)

        for new_eval in results:
            new_eval["labels_file"] = label_file 
            summary.add_evaluation(new_eval, checkpoint_key=eval_idx)
            eval_idx += 1
    
    summary.save()
    print(summary)

