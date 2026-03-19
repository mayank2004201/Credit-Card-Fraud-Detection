import os
import shutil
from fraud_detection import logger
from fraud_detection.entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Copies the files from train_source_URL and test_source_URL to local artifacts directory.
        """
        for source, filename in [(self.config.train_source_URL, "fraudTrain.csv"), 
                                (self.config.test_source_URL, "fraudTest.csv")]:
            if not os.path.exists(source):
                logger.error(f"Source file not found at {source}")
                continue

            if not os.path.exists(os.path.join(self.config.root_dir, filename)):
                shutil.copy(source, self.config.root_dir)
                logger.info(f"{filename} copied to {self.config.root_dir}")
            else:
                logger.info(f"{filename} already exists in {self.config.root_dir}")

    def extract_zip_file(self):
        """
        In this case, we have a CSV, so this might just be a placeholder 
        or a simple existence check.
        """
        pass
