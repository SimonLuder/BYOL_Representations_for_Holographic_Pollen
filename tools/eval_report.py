import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import scipy.stats as stats
from ssl_poleno.evaluation.report import EvaluationSummary
from ssl_poleno.evaluation.mrr import calc_mrr_pd
from ssl_poleno.evaluation.utils import load_inference_results, limit_events_per_species

from ssl_poleno.evaluation import knn


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


def run_tests(
        checkpoints, 
        labels, 
        embeddings=None, 
        ckpt_root="checkpoints", 
        k_fold = 5, 
        k_neighbors=10, 
        train_sizes=[100],
        mrr_max_events=50,
    ):

    # Load ground truth labels
    test_labels = pd.read_csv(labels)

    results = []
    for checkpoint in checkpoints:

        if embeddings is None:
            embeddings = find_files_by_regex(
                root_dir=os.path.join(ckpt_root, checkpoint), 
                pattern=r"inference_.*\.npz"
            )

        for embedding in embeddings:

            filename = os.path.join(ckpt_root, checkpoint, embedding)
            pred_dir = os.path.join(ckpt_root, checkpoint, "predictions")
            print(f"Calculate respresentations for file: {filename}")

            representations, labels_name = load_inference_results(filename)

            df = pd.merge(representations, test_labels, on="rec_path", how='inner')
            df = df.sort_values(["species", "event_id"]).reset_index(drop=True)

            for train_size in train_sizes:

                out = knn.evaluate_embeddings_knn_cv(
                    df, 
                    y_col="species", 
                    k=k_neighbors, 
                    n_splits=k_fold,
                    train_samples_per_class=train_size,
                )

                predictions, true_labels, test_indices, accuracies = out

                # Flatten CV folds
                predictions = np.concatenate(predictions)
                true_labels = np.concatenate(true_labels)
                flat_indices = np.concatenate(test_indices)

                mean_acc, ci = calc_cv_accuracy_ci(np.array(accuracies))
                acc_se, acc_sd = calc_sd_se(np.array(accuracies))

                df_mrr = limit_events_per_species(
                    df,
                    species_col="species",
                    event_col="event_id",
                    max_events=mrr_max_events
                )
                event_mrr = calc_mrr_pd(df_mrr, emb_col="emb", lbl_col="event_id")


                # Save predictions
                test_meta = df.iloc[test_indices][["species", "event_id", "rec_path"]].reset_index(drop=True)
                os.makedirs(pred_dir, exist_ok=True)
                pred_file = os.path.join(pred_dir, f"{version}_train{train_size}.npz")

                np.savez(
                    pred_file,
                    predictions=predictions,
                    true_labels=true_labels,
                    test_indices=test_indices,
                    accuracies=np.array(accuracies),
                    species=test_meta["species"].values,
                    event_ids=test_meta["event_id"].values,
                    rec_paths=test_meta["rec_path"].values,
                )

                version = os.path.basename(os.path.dirname(filename))

                result = {
                    "checkpoint": checkpoint,
                    "version": version,
                    "labels": labels_name,

                    "train_samples_per_class": train_size,

                    "mean_cv_accuracy": mean_acc,
                    "cv_accuracy_ci_low_95": ci[0],
                    "cv_accuracy_ci_high_95": ci[1],
                    "cv_accuracy_sd": acc_sd,
                    "cv_accuracy_se": acc_se,

                    "k_fold": k_fold,
                    "k_neighbours": k_neighbors,

                    "event_mrr": event_mrr,
                    "mrr_max_events": mrr_max_events,
                    "mrr_total_events": len(df_mrr),

                    "predictions_file": pred_file,
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
    
    parser = argparse.ArgumentParser(description='Arguments for postprocessing')

    parser.add_argument(
        '--outfile',  
        default="evaluation_summary_abl.csv", 
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
        default=["inference/knn/inference_basic_test.npz"], 
        type=str, 
        nargs='+', 
        help='List of embeddings files'
    )

    parser.add_argument(
        '--max_train_size', 
        default=[100,], 
        type=int, 
        nargs='+', 
        help='Max nr of reference labels per species for knn'
    )
    
    args = parser.parse_args()

    summary = EvaluationSummary(args.outfile, overwrite=True)

    checkpoints = args.ckpt_names

    eval_idx = 0
    for label_file in args.labels:
        results = run_tests(
            checkpoints, 
            label_file,
            args.embeddings, 
            train_sizes=args.max_train_size,
        )

        for new_eval in results:
            new_eval["labels_file"] = label_file 
            summary.add_evaluation(new_eval, checkpoint_key=eval_idx)
            eval_idx += 1
    
    summary.save()
    print(summary)

