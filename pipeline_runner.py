#!/usr/bin/env python3
"""
Pipeline Controller
===================

This script serves as the main entry point for the machine learning pipeline.
It handles configuration parsing, environment setup (logging, seeding), and
orchestrates the data preparation, model training, and inference stages.

Modes:
    - Application: Standard train/test split execution (Single Run).
    - Evaluation: Supports single-run or batch-run evaluation workflows.
"""

import argparse
import json
import sys
import shutil
import logging
import time
import random
from pathlib import Path

import numpy as np
import torch

from pipeline_utils.data_manager import DataManager
from pipeline_utils.model_builder import ModelBuilder
from pipeline_utils.train_engine import TrainEngine
from pipeline_utils.inference_engine import InferenceEngine


# ==========================================
# Logging & Utility Functions
# ==========================================

def setup_logging(output_dir: Path):
    """Configures logging to both file and console."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []

    log_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log_fmt.converter = time.localtime

    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    file_handler.setFormatter(log_fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_fmt)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def log_section(title: str):
    """Helper to print formatted section headers in logs."""
    logging.info("\n" + "="*60)
    logging.info(f" {title.upper()} ")
    logging.info("="*60)

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Main Controller
# ==========================================

class PipelineController:
    """
    Manages the end-to-end execution of the pipeline based on a JSON configuration.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        # Load Configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup Output Directory Structure
        self.project_name = self.config['project_name'].replace(" ", "_")
        self.output_dir = Path(f"results/{self.project_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)

        # Initialize Environment
        setup_logging(self.output_dir)
        seed = self.config.get('model_architecture', {}).get('random_seed', 42)
        set_seed(seed)

        # Archive Config
        shutil.copy(self.config_path, self.output_dir / "config_snapshot.json")

    def run(self):
        """
        Determines the execution mode (Application vs Evaluation) and 
        triggers the appropriate pipeline workflow.
        """
        mode = self.config.get('mode', 'application')
        
        try:
            if mode == 'application':
                # In application mode, we expect app_config with train_path/test_path
                app_config = self.config.get('app_config') or {}
                if not app_config:
                    raise ValueError("Mode is 'application' but 'app_config' is missing or empty.")
                
                # Robustness Fix: Explicit validation of required keys
                train_path = app_config.get('train_path')
                test_path = app_config.get('test_path')
                
                if not train_path or not test_path:
                    raise KeyError("Mode is 'application' but 'train_path' or 'test_path' is missing from app_config.")

                self.run_pipeline(train_path, test_path, "app_run")
            
            elif mode == 'evaluation':
                # Robustness Fix: Handle case where eval_config is explictly null in JSON
                eval_config = self.config.get('eval_config') or {}
                method = eval_config.get('method', 'single')
                
                if method == 'single':
                    logging.info("Starting Single Evaluation Run...")
                    
                    train_path = eval_config.get('train_path')
                    test_path = eval_config.get('test_path')
                    
                    if not train_path or not test_path:
                        raise KeyError("Evaluation method is 'single' but 'train_path' or 'test_path' is missing from JSON.")

                    self.run_pipeline(train_path, test_path, "eval_single")
                
                elif method == 'batch':
                    logging.info("Starting Batch Evaluation Run...")
                    
                    folder_path_str = eval_config.get('folder_path', 'splits')
                    folder_path = Path(folder_path_str)
                    
                    if not folder_path.exists():
                        raise FileNotFoundError(f"Batch folder not found: {folder_path.resolve()}")
                    
                    # Dynamic File Discovery: Find all files ending in '_test.csv'
                    test_files = sorted(list(folder_path.glob("*_test.csv")))
                    
                    if not test_files:
                        all_csvs = list(folder_path.glob("*.csv"))
                        if not all_csvs:
                             raise FileNotFoundError(f"No CSV files found in: {folder_path}")
                        else:
                             raise FileNotFoundError(f"Found CSVs, but none matched pattern '*_test.csv' in: {folder_path}")
                    
                    logging.info(f"Found {len(test_files)} test files to process.")
                    
                    for i, test_file in enumerate(test_files):
                        # Construct corresponding train file name
                        # Assumes syntax: "Name_test.csv" -> "Name_train.csv"
                        test_name = test_file.name
                        base_name = test_name.replace("_test.csv", "")
                        train_name = f"{base_name}_train.csv"
                        train_file = folder_path / train_name
                        
                        task_id = f"batch_{base_name}"
                        
                        if not train_file.exists():
                            logging.warning(f"Skipping {test_name}: Corresponding train file {train_name} not found.")
                            continue

                        logging.info(f"Processing Batch {i+1}/{len(test_files)}: {task_id}")
                        logging.info(f"  > Train: {train_file.name}")
                        logging.info(f"  > Test:  {test_file.name}")
                        
                        self.run_pipeline(str(train_file), str(test_file), task_id)

        except Exception as e:
            logging.exception(f"Critical failure: {e}")
            sys.exit(1)

    def run_pipeline(self, train_csv: str, test_csv: str, task_id: str):
        """
        Executes the standard four-step pipeline: Data -> Model -> Train -> Inference.

        Args:
            train_csv (str): Path to training data.
            test_csv (str): Path to testing data.
            task_id (str): Unique identifier for this run (used for file naming).
        """
        
        # 1. Data Preparation & Featurization
        log_section(f"1. Data & Features ({task_id})")
        
        dm = DataManager(
            self.config['data'], 
            self.output_dir / "features", 
            self.config['task']
        )
        
        bs = int(self.config['training'].get('batch_size', 32))
        loaders, dims, task_map = dm.prepare_data(train_csv, test_csv, batch_size=bs)
        
        logging.info(f"Input Feature Dim: {dims['features']}")
        logging.info(f"Output Label Dim: {dims['output_dim']}")

        # 2. Model Initialization
        log_section("2. Model Architecture")
        
        mb = ModelBuilder(self.config['model_architecture'])
        model = mb.build(
            input_dim=dims['features'], 
            task_structure=task_map, 
            output_dim=dims['output_dim']
        )

        # 3. Model Training
        log_section("3. Training")
        
        trainer = TrainEngine(
            self.config['training'], 
            self.output_dir / "models", 
            task_id, 
            self.config['task']
        )
        best_model_path = trainer.execute(model, loaders, task_map)
        
        # 4. Inference / Testing
        log_section("4. Inference")
        
        inference = InferenceEngine(
            self.config['model_architecture'], 
            self.output_dir / "predictions",
            task_config=self.config['task'],
            label_map=dm.label_map
        )
        inference.run(model, best_model_path, loaders['test'], task_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Pipeline")
    parser.add_argument('--config', required=True, help="Path to the JSON configuration file")
    args = parser.parse_args()
    
    PipelineController(args.config).run()