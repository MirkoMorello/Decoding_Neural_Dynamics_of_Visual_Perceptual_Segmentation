import pandas as pd
import os
from pathlib import Path
import argparse
from collections import defaultdict
import re

def get_best_epoch_metrics(csv_file):
    """
    Reads a progress_log.csv file and returns the metrics from the epoch
    with the best validation Log-Likelihood.
    """
    try:
        df = pd.read_csv(csv_file)
        df.columns = [col.strip() for col in df.columns]
        if 'validation_LL' not in df.columns or df['validation_LL'].isnull().all():
            return None
        best_epoch_idx = df['validation_LL'].idxmax()
        best_epoch_series = df.loc[best_epoch_idx]
        metrics_to_extract = ['validation_LL', 'validation_IG', 'validation_NSS', 'validation_AUC_CPU']
        if not all(metric in best_epoch_series.index for metric in metrics_to_extract):
            return None
        return best_epoch_series[metrics_to_extract]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        return None

def main(experiments_base_dir):
    base_path = Path(experiments_base_dir)
    
    stage_patterns = [
        "salicon_pretrain*",
        "mit_spatial_finetune*",
        "mit_scanpath_frozen*"
    ]

    print("--- Final Experiment Statistics (Grouped by Stage and Experiment) ---")

    for stage_pattern in stage_patterns:
        stage_dirs = [d for d in base_path.glob(stage_pattern) if d.is_dir()]
        if not stage_dirs:
            continue

        for stage_dir in stage_dirs:
            print(f"\n\n--- Stage: {stage_dir.name} ---")
            
            experiments_in_stage = defaultdict(list)
            run_dirs = [d for d in stage_dir.iterdir() if d.is_dir()]

            for run_dir in run_dirs:
                if 'salicon' in stage_dir.name.lower():
                    exp_base_name = run_dir.name
                else:
                    exp_base_name = re.split(r'_fold\d+', run_dir.name)[0]
                
                log_file = run_dir / 'progress_log.csv'
                if log_file.exists():
                    experiments_in_stage[exp_base_name].append(log_file)

            if not experiments_in_stage:
                print("  No experiment logs found in this stage directory.")
                continue

            for exp_name in sorted(experiments_in_stage.keys()):
                log_files = experiments_in_stage[exp_name]
                print(f"\n  --- Experiment: {exp_name} ---")
                
                all_best_metrics = []
                for log_file in log_files:
                    best_metrics = get_best_epoch_metrics(log_file)
                    if best_metrics is not None:
                        all_best_metrics.append(best_metrics)

                if not all_best_metrics:
                    print("    Could not extract valid metrics from any log file for this experiment.")
                    continue

                results_df = pd.DataFrame(all_best_metrics).astype(float)
                summary_stats = results_df.agg(['mean', 'std'])

                print(f"    Found {len(results_df)} successful runs/folds.")
                print("    Averaged results (from best validation LL epoch):")
                print(summary_stats.to_string(float_format="%.4f").replace('\n', '\n    '))
                print("  " + "-" * (len(exp_name) + 18))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and print final statistics from experiment logs, grouped by stage and experiment name.")
    parser.add_argument(
        '--path',
        type=str,
        default='./experiments',
        help='Path to the base experiments directory.'
    )
    args = parser.parse_args()
    
    main(args.path)
