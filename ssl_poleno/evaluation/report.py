import os
import pandas as pd
from pathlib import Path


class EvaluationSummary:
    def __init__(self, summary_file: str, overwrite: bool = False):
        self.summary_file = summary_file
        self.eval_df = self._load_existing(overwrite)

    def _load_existing(self, overwrite) -> pd.DataFrame:
        """Load existing evaluation summary if it exists."""
        if overwrite:
            return pd.DataFrame()
        
        if os.path.isfile(self.summary_file):
            return pd.read_csv(self.summary_file)
        
        return pd.DataFrame()

    def add_evaluation(self, run_evaluation: dict, checkpoint_key: str = "checkpoint"):
        """
        Add a new evaluation to the summary.
        Removes any existing rows with the same checkpoint.
        """
        checkpoint_value = run_evaluation.get(checkpoint_key)

        if checkpoint_value is not None and not self.eval_df.empty:
            self.eval_df = self.eval_df[
                self.eval_df[checkpoint_key] != checkpoint_value
            ]

        run_eval_df = pd.DataFrame([run_evaluation])
        self.eval_df = pd.concat(
            [self.eval_df, run_eval_df], ignore_index=True
        )

    def save(self):
        """Persist the evaluation summary to disk."""
        Path(self.summary_file).parent.mkdir(parents=True, exist_ok=True)
        self.eval_df.to_csv(self.summary_file, index=False)

    def __repr__(self):
        return repr(self.eval_df)
    

