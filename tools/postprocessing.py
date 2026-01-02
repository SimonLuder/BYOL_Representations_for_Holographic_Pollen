import os
from .inference import inference
from byol_poleno.utils.pipeline import extract_backbone_state_dict_from_lightning_ckpt, get_best_checkpoint_by_loss
from byol_poleno.utils.config import get_ckpt_config_file


def run_pipeline(force_run=False, localpath="checkpoints/"):

    for model in checkpoint_names:

        ckpt_folder = os.path.join(localpath, model)
        best_ckpt = get_best_checkpoint_by_loss(ckpt_folder)

        all_inference_ckpts = {
            "best": os.path.join(localpath, model, best_ckpt),
            "last": os.path.join(localpath, model, "last.ckpt"),
        }

        for ckpt_name, ckpt_path in all_inference_ckpts.items():

            # Extract backbone state_dict from Lightning checkpoint
            state_dict = os.path.join(localpath, model, f"backbone_{ckpt_name}.pt")
            if not os.path.exists(state_dict) or force_run:
                extract_backbone_state_dict_from_lightning_ckpt(
                    ckpt_path,
                    save_path=state_dict,
                )
            else:
                print(f"{model}: Backbone state_dict exists. Skipping extraction.")

            # Run inference to generate embeddings if not already done
            inference_file = os.path.join(localpath, model, f"inference_{ckpt_name}.npz")
            if not os.path.exists(inference_file) or force_run:
                print(f"Running inference for model: {model}")

                config_path = get_ckpt_config_file(ckpt_path)
                print("config_path", config_path)
                inference(
                    ckpt_path=ckpt_path,
                    config_path=config_path,
                    save_as=f"inference_{ckpt_name}.npz",
                    config_updates=config_updates
                )
            else:
                print(f"{model}: Embeddings already exist. Skipping inference.")


if __name__ == "__main__":

    checkpoint_names = [
        "byol_lightning_20251202_121331",
        "byol_lightning_20251202_121258",
        "byol_lightning_20251202_120917",
        "byol_lightning_20251219_151741",
        "byol_lightning_20251219_143015",
        "byol_lightning_20251223_143929",
        "byol_lightning_20251223_144036",
        "byol_lightning_20251230_114203",
    ]

    config_updates = {
    "dataset": {
        "root": "Z:/marvel/marvel-fhnw/data/",
        "labels_train": "data/final/poleno/basic_train.csv",
        "labels_val": "data/final/poleno/basic_val_20.csv",
        "labels_test": "data/final/poleno/basic_test_20.csv",
        }
    }   

    run_pipeline(force_run=False)