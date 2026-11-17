"""
EV Battery Health Monitoring - Main Pipeline Runner
Orchestrates the complete end-to-end ML pipeline.
"""
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
DASHBOARD_DIR = ROOT_DIR / "dashboard"


def run_script(script_name: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT_DIR, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        return False


def run_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = DASHBOARD_DIR / "app.py"
    if not dashboard_path.exists():
        print(f"Dashboard not found: {dashboard_path}")
        return
    
    print(f"\n{'='*60}")
    print("Launching Streamlit Dashboard...")
    print(f"{'='*60}")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], cwd=ROOT_DIR)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def main():
    """Main entry point for the EV Battery Health Monitoring pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EV Battery Health Monitoring Pipeline")
    parser.add_argument(
        "stage",
        choices=["preprocess", "features", "train", "dashboard", "all", "test"],
        help="Pipeline stage to run"
    )
    parser.add_argument("--dashboard-only", action="store_true", help="Launch dashboard only")
    
    args = parser.parse_args()
    
    if args.dashboard_only or args.stage == "dashboard":
        run_dashboard()
        return
    
    if args.stage == "all" or args.stage == "preprocess":
        print("\n🔄 Stage 1: Data Preprocessing")
        if not run_script("data_preprocessing.py"):
            print("❌ Preprocessing failed!")
            return
        print("✅ Preprocessing complete!")
    
    if args.stage == "all" or args.stage == "features":
        print("\n🔄 Stage 2: Feature Engineering")
        if not run_script("feature_engineering.py"):
            print("❌ Feature engineering failed!")
            return
        print("✅ Feature engineering complete!")
    
    if args.stage == "all" or args.stage == "train":
        print("\n🔄 Stage 3: Model Training & Evaluation")
        if not run_script("model_training.py"):
            print("❌ Model training failed!")
            return
        print("✅ Model training complete!")
    
    if args.stage == "test":
        print("\n🔄 Running Maintenance Decision Engine Tests")
        if not run_script("maintenance_decision.py"):
            print("❌ Tests failed!")
            return
        print("✅ Tests complete!")
    
    if args.stage == "all":
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Launch dashboard: python run.py dashboard")
        print("  2. View results in models/ and plots/ directories")
        print("  3. Check model_report.txt for detailed evaluation")


if __name__ == "__main__":
    main()