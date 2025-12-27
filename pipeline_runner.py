#!/usr/bin/env python3
import argparse
import json
import sys
import shutil
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np
import torch

from pipeline_utils.data_manager import DataManager
from pipeline_utils.model_builder import ModelBuilder
from pipeline_utils.train_engine import TrainEngine
from pipeline_utils.inference_engine import InferenceEngine

# ... (Logging and Seed Utils same as before) ...
class LocalFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt: s = time.strftime(datefmt, ct)
        else: t = time.strftime("%Y-%m-%d %H:%M:%S", ct); s = "%s,%03d" % (t, record.msecs)
        return s

def setup_logging(output_dir: Path):
    root_logger = logging.getLogger()
    if root_logger.handlers: root_logger.handlers = []
    log_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_fmt.converter = time.localtime
    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    file_handler.setFormatter(log_fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_fmt)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def log_section(title):
    logging.info("\n" + "="*60); logging.info(f" {title.upper()} "); logging.info("="*60)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PipelineController:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f: self.config = json.load(f)
        self.project_name = self.config['project_name'].replace(" ", "_")
        self.output_dir = Path(f"results/{self.project_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        setup_logging(self.output_dir)
        seed = self.config.get('model_architecture', {}).get('random_seed', 42)
        set_seed(seed)
        shutil.copy(self.config_path, self.output_dir / "config_snapshot.json")

    def run(self):
        mode = self.config.get('mode', 'application')
        try:
            if mode == 'application':
                self.run_pipeline(self.config['app_config']['train_path'], self.config['app_config']['test_path'], "app_run")
            elif mode == 'evaluation':
                # (Eval logic stub - assumes same structure)
                eval_config = self.config['eval_config']
                if eval_config['method'] == 'single':
                    self.run_pipeline(eval_config['train_path'], eval_config['test_path'], "eval_single")
                # ... (Batch logic omitted for brevity, same as V3)
        except Exception as e:
            logging.exception(f"Critical failure: {e}"); sys.exit(1)

    def run_pipeline(self, train_csv: str, test_csv: str, task_id: str):
        # 1. DATA PREP
        log_section("1. Data & Features")
        # Pass holistic task config to DataManager
        dm = DataManager(self.config['data'], self.output_dir / "features", self.config['task'])
        
        bs = int(self.config['training'].get('batch_size', 32))
        loaders, dims, task_map = dm.prepare_data(train_csv, test_csv, batch_size=bs)
        
        logging.info(f"Input Feature Dim: {dims['features']}")
        logging.info(f"Output Label Dim: {dims['output_dim']}")

        # 2. MODEL BUILDING
        log_section("2. Model Architecture")
        mb = ModelBuilder(self.config['model_architecture'])
        # Pass the detected output dimension (e.g. 500 DDI tags) to the builder
        model = mb.build(input_dim=dims['features'], task_structure=task_map, output_dim=dims['output_dim'])

        # 3. TRAINING
        log_section("3. Training")
        # Pass holistic task config to Trainer
        trainer = TrainEngine(self.config['training'], self.output_dir / "models", task_id, self.config['task'])
        best_model_path = trainer.execute(model, loaders, task_map)
        
        # 4. INFERENCE
        log_section("4. Inference")
        inference = InferenceEngine(
            self.config['model_architecture'], 
            self.output_dir / "predictions",
            task_config=self.config['task'], # Pass task config
            label_map=dm.label_map         # Pass the label map detected by DataManager
        )
        inference.run(model, best_model_path, loaders['test'], task_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    PipelineController(args.config).run()