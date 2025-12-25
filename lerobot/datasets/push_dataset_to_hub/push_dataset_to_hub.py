#!/usr/bin/env python
"""
Push a local LeRobot dataset to Hugging Face Hub.

Usage:
    python push_dataset_to_hub.py --dataset-dir outputs/expert_data/oracle_reach_v3_100 --repo-id username/dataset_name
"""

import argparse
import logging
from pathlib import Path
import os
from huggingface_hub import HfApi, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Push LeRobot dataset to Hugging Face Hub")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the local dataset directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/dataset_name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on Hugging Face Hub",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, will use cached token)",
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    
    logger.info(f"Loading dataset from {dataset_dir}")
    
    try:
        # Create API instance
        api = HfApi()

        # Create repository if it doesn't exist
        logger.info(f"Creating repository {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            token=args.token,
            exist_ok=True
        )

        # Upload the entire dataset folder
        logger.info(f"Uploading dataset from {dataset_dir} to {args.repo_id}")
        upload_folder(
            repo_id=args.repo_id,
            folder_path=str(dataset_dir),
            repo_type="dataset",
            token=args.token,
        )

        logger.info(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.repo_id}")

    except Exception as e:
        logger.error(f"Failed to push dataset: {e}")
        raise


if __name__ == "__main__":
    main()
