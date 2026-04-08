"""
=============================================================
  JobSuit — Main Application Launcher
=============================================================
  Usage:
    python fake_job_detector.py --mode api     # Start the JobSuit Web Dashboard
    python fake_job_detector.py --mode train   # Retrain the Machine Learning Model
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="JobSuit — AI Career Intelligence & Fraud Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["api", "train"],
        default="api",
        help=(
            "api    — Launch the JobSuit Web Dashboard (Default)\n"
            "train  — Retrain the ML model on dataset"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/jobs.csv",
        help="Path to training CSV dataset (used with --mode train)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Directory to save/load model artifacts",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask API server",
    )

    args = parser.parse_args()

    # Safely set encoding to prevent terminal emoji crashes on Windows
    os.environ["PYTHONIOENCODING"] = "utf-8"

    if args.mode == "train":
        print(f"🧠 Initiating Model Training Pipeline on {args.data}...")
        from src.pipeline import train_pipeline
        train_pipeline(data_path=args.data, model_dir=args.model_dir)

    elif args.mode == "api":
        from src.api import create_app
        app = create_app(model_dir=args.model_dir)
        print(f"\n🚀 JobSuit Dashboard running at http://127.0.0.1:{args.port}\n")
        app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
