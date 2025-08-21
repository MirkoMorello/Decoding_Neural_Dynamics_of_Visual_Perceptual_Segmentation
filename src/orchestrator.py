# orchestrator.py
import yaml
import subprocess
import argparse
import logging
from pathlib import Path
import copy
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

    # --- Real-time output streaming and filtering ---
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    error_keywords = ["Traceback", "Error", "Exception", "FATAL", "failed"]

    for line in iter(process.stdout.readline, ''):
        if nproc_per_node > 1:
            is_error_line = any(keyword.lower() in line.lower() for keyword in error_keywords)
            if "[RANK 0]" in line or is_error_line:
                sys.stdout.write(line)
        else:
            sys.stdout.write(line)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    
    logging.info(f"STAGE COMPLETED: {stage_name}")

def main():
    """Top-level function to run experiments defined in a master config."""
    parser = argparse.ArgumentParser(description="Orchestrator for multi-stage gaze prediction experiments.")
    parser.add_argument("--master-config", type=str, default="configs/pipeline.yaml", help="Path to the master YAML file listing all experiments to run.")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs/processes. >1 enables DDP.")
    args = parser.parse_args()

    with open(args.master_config, 'r') as f:
        master_config = yaml.safe_load(f)

    # This dictionary will store the output paths of completed stages
    checkpoint_store = {}

    for stage in master_config['sequence']:
        stage_type = stage['type']
        stage_name = stage['name']
        config_path = stage['config_path']
        
        # Load the base name from the config file to construct fold names
        with open(config_path, 'r') as f:
             base_stage_name = yaml.safe_load(f).get('stage', {}).get('name', 'base_name')

        if stage_type == 'single':
            overrides = {'stage.name': base_stage_name}
            run_stage(config_path, overrides, args.nproc_per_node)
            checkpoint_store[stage_name] = get_output_checkpoint_path(config_path, overrides)

        elif stage_type == 'loop':
            input_stage_name = stage['input_checkpoint_from']
            base_input_ckpt_path = checkpoint_store.get(input_stage_name)
            if not base_input_ckpt_path:
                raise ValueError(f"Checkpoint for input stage '{input_stage_name}' not found. Ensure it runs before this stage.")

            for fold in range(stage['num_folds']):
                fold_name = f"{base_stage_name}_fold{fold}"
                overrides = {
                    'stage.name': fold_name,
                    'stage.extra.fold': fold
                }

                # Determine the correct input checkpoint for this fold
                if input_stage_name == 'mit_spatial_finetune':
                    # Scanpath depends on the spatial stage of the *same fold*
                    spatial_fold_overrides = {'stage.name': f"{base_stage_name.replace('scanpath_frozen', 'spatial_finetune')}_fold{fold}"}
                    # We need the config path of the spatial stage to calculate this
                    spatial_config_path = next(s['config_path'] for s in master_config['sequence'] if s['name'] == 'mit_spatial_finetune')
                    overrides['stage.resume_ckpt'] = get_output_checkpoint_path(spatial_config_path, spatial_fold_overrides)
                else:
                    # Spatial always depends on the single pre-train stage
                    overrides['stage.resume_ckpt'] = base_input_ckpt_path

                run_stage(config_path, overrides, args.nproc_per_node)
                # We don't strictly need to store loop checkpoints unless another stage depends on them
        
        else:
             logging.error(f"Unknown stage type '{stage_type}' in master config.")

    logging.info("Orchestration of all stages has finished successfully.")

if __name__ == "__main__":
    main()