import os
import argparse
from .inference import inference
from ssl_poleno.utils import pipeline
from ssl_poleno.utils import config


def run_pipeline(checkpoint_names, force_run=False, localpath="checkpoints/", config_updates=None):

    for model in checkpoint_names:

        ckpt_folder = os.path.join(localpath, model)
        best_ckpt = pipeline.get_best_checkpoint_by_val_loss(ckpt_folder)
        best_knn_ckpt = pipeline.get_best_checkpoint_by_val_knn_acc(ckpt_folder)
        best_mrr_ckpt = pipeline.get_best_checkpoint_by_mrr(ckpt_folder)

        all_inference_ckpts = {
            "best": os.path.join(localpath, model, best_ckpt),
            "last": os.path.join(localpath, model, "last.ckpt"),
        }

        if best_knn_ckpt is not None:
            all_inference_ckpts["knn"] = os.path.join(localpath, model, best_knn_ckpt)

        if best_mrr_ckpt is not None:
            all_inference_ckpts["mrr"] = os.path.join(localpath, model, best_mrr_ckpt)

        for ckpt_name, ckpt_path in all_inference_ckpts.items():

            print(ckpt_name, ckpt_path)

            # Extract backbone state_dict from Lightning checkpoint
            state_dict = os.path.join(localpath, model, f"backbone_{ckpt_name}.pt")
            if not os.path.exists(state_dict) or force_run:
                pipeline.extract_backbone_state_dict_from_lightning_ckpt(
                    ckpt_path,
                    save_path=state_dict,
                )
            else:
                print(f"{model}: Backbone state_dict exists. Skipping extraction.")

            # Run inference to generate embeddings if not already done
            config_path = config.get_ckpt_config_file(ckpt_path)
            cfg = config.load(config_path)
            if config_updates is not None:
                cfg = config.deep_update(cfg, config_updates)

            test_labels = cfg["dataset"]["labels_test"]
            test_name = os.path.splitext(os.path.basename(test_labels))[0]
            save_as = os.path.join("inference", ckpt_name, f"inference_{test_name}.npz")
            inference_file = os.path.join(localpath, model, save_as)

            if not os.path.exists(inference_file) or force_run:
                os.makedirs(os.path.dirname(inference_file), exist_ok=True)
                print(f"Running inference for model: {model}")
                print("Using config_path", config_path)
                inference(
                    ckpt_path=ckpt_path,
                    conf=cfg,
                    save_as=save_as,
                )
            else:
                print(f"{model}: Embeddings already exist. Skipping inference.")


if __name__ == "__main__":

    checkpoint_names = [
        "byol_lit_20260204_152517",
    ]

    config_updates = {
    "dataset": {
        "root": "Z:/marvel/marvel-fhnw/data/",
        "labels_test": "data/final/poleno/isolated_test_20.csv",
        }
    }   

    parser = argparse.ArgumentParser(description='Arguments for postprocessing')
    parser.add_argument('--localpath', dest='localpath', default='checkpoints/', type=str)
    parser.add_argument('--names', dest='ckpt_names', nargs='+', default=None, 
                        type=str, help='List of checkpoint / model names')
    args = parser.parse_args()

    if args.ckpt_names is not None:
        checkpoint_names = args.ckpt_names

    run_pipeline(checkpoint_names, 
                 localpath=args.localpath, 
                 config_updates=config_updates, 
                 force_run=False)