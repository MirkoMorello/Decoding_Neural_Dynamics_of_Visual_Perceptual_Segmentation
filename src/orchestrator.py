# orchestrator.py
import yaml
import subprocess
import argparse
import logging
from pathlib import Path
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][ORCHESTRATOR][%(levelname)s] - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def get_output_checkpoint_path(config_path: str, overrides: dict) -> Path:
    """
    Calculates the predictable output checkpoint path by reading the original config
    and applying any name overrides.
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get base values from the config file
    base_train_dir = Path(config_data.get('paths', {}).get('train_dir', './experiments'))
    kind = config_data.get('stage', {}).get('kind', 'unknown_kind')
    
    # The final name is determined by the override, if it exists
    name = overrides.get('stage.name', config_data.get('stage', {}).get('name', 'unknown_name'))
    
    return base_train_dir / kind / name / "final_best_val.pth"

def run_stage(config_path: str, overrides: dict, nproc_per_node: int):
    """
    Runs a single training stage using a specific config file and command-line overrides.
    Streams output in real-time, filtering for Rank 0 messages in DDP mode.
    """
    stage_name = overrides.get('stage.name', 'N/A')
    final_checkpoint_path = get_output_checkpoint_path(config_path, overrides)
    if final_checkpoint_path.exists():
        logging.info(f"--- SKIPPING STAGE: {stage_name} ---")
        logging.info(f"Final checkpoint already exists at: {final_checkpoint_path}")
        return # Exit the function immediately
    logging.info("="*80)
    logging.info(f"STARTING STAGE: {stage_name}")
    logging.info(f"Using base config: {config_path}")
    logging.info(f"Running with {nproc_per_node} GPU(s).")
    logging.info(f"Overrides: {overrides}")
    logging.info("="*80)

    # --- Construct the command ---
    if nproc_per_node > 1:
        command = ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}", "-m", "src.train"]
    else:
        command = ["python", "-m", "src.train"]
    
    command.extend(["--config", config_path])
    
    # Append overrides as key=value pairs
    for key, value in overrides.items():
        if value is not None:
            command.append(f"{key}={value}")

    logging.info(f"Executing command: {' '.join(command)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" # Ensure real-time output for tqdm compatibility

    # By default, subprocess.run connects the child's stdout/stderr to the parent's.
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Stage '{stage_name}' failed with exit code {e.returncode}.")
        raise # Re-raise the exception to stop the orchestration
    
    logging.info(f"STAGE COMPLETED: {stage_name}")

def main():
    """Top-level function to run experiments defined in a master config."""
    parser = argparse.ArgumentParser(description="Orchestrator for multi-stage gaze prediction experiments.")
    parser.add_argument("--master-config", type=str, default="configs/v2/pipeline.yaml", help="Path to the master YAML file listing all experiments to run.")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs/processes. >1 enables DDP.")
    args = parser.parse_args()

    with open(args.master_config, 'r') as f:
        master_config = yaml.safe_load(f)

    # This will only hold the single, static checkpoint path from the pre-training stage.
    salicon_checkpoint_path = None

    # Get the base train directory once from the first config listed.
    # Assumes all stages use the same top-level experiment directory.
    first_config_path = master_config['sequence'][0]['config_path']
    with open(first_config_path, 'r') as f:
        base_train_dir = Path(yaml.safe_load(f).get('paths', {}).get('train_dir', './experiments'))

    for stage in master_config['sequence']:
        stage_type = stage['type']
        config_path = stage['config_path']
        
        # Load the kind and base_name from the specific config file for this stage.
        with open(config_path, 'r') as f:
             stage_config = yaml.safe_load(f).get('stage', {})
             stage_kind = stage_config.get('kind')
             base_stage_name = stage_config.get('name')

        if stage_type == 'single':
            overrides = {'stage.name': base_stage_name}
            run_stage(config_path, overrides, args.nproc_per_node)
            
            # Store the path of the pre-training checkpoint.
            salicon_checkpoint_path = get_output_checkpoint_path(config_path, overrides)

        elif stage_type == 'loop':
            input_stage_name = stage['input_checkpoint_from']
            
            for fold in range(stage['num_folds']):
                fold_name = f"{base_stage_name}_fold{fold}"
                overrides = {
                    'stage.name': fold_name,
                    'stage.extra.fold': fold
                }

                if input_stage_name == 'mit_spatial_finetune':
                    spatial_kind = 'mit_spatial_finetune'
                    spatial_base_name = base_stage_name.replace('scanpath_frozen', 'spatial_finetune')
                    spatial_fold_name = f"{spatial_base_name}_fold{fold}"
                    
                    # The path is simply: <train_dir>/<spatial_kind>/<spatial_fold_name>/final_best_val.pth
                    input_ckpt_path = base_train_dir / spatial_kind / spatial_fold_name / 'final_best_val.pth'
                    
                    overrides['stage.resume_ckpt'] = str(input_ckpt_path)

                else: # This is the spatial stage, depending on the salicon pre-train.
                    if not salicon_checkpoint_path:
                        raise ValueError("SALICON pre-training must be run before the MIT stages.")
                    overrides['stage.resume_ckpt'] = str(salicon_checkpoint_path)
                
                run_stage(config_path, overrides, args.nproc_per_node)
        
        else:
             logging.error(f"Unknown stage type '{stage_type}' in master config.")

    logging.info("Orchestration of all stages has finished successfully.")

if __name__ == "__main__":
    main()