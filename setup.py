"""
Setup script for Sales Forecasting Project
Run this to verify your environment is set up correctly
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_directories():
    """Check if required directories exist"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'reports',
        'src',
        'dashboard',
        'api'
    ]

    all_exist = True
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
            all_exist = False

    return all_exist


def check_dependencies():
    """Check if required packages are installed"""

    # pip_name : import_name
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',   # IMPORTANT
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'fastapi': 'fastapi'
    }

    missing = []

    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name} installed")
        except ImportError:
            print(f"❌ {pip_name} NOT installed")
            missing.append(pip_name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_data():
    """Check if sample data exists"""
    data_path = Path('data/raw/sales_data.csv')
    if data_path.exists():
        print(f"✅ Sample data found: {data_path}")
        return True
    else:
        print("⚠️  Sample data not found.")
        print("Run: python src/data/generate_sample_data.py")
        return False


def main():
    """Run all checks"""
    print("=" * 50)
    print("Sales Forecasting Project - Setup Check")
    print("=" * 50)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Directories", check_directories),
        ("Dependencies", check_dependencies),
        ("Sample Data", check_data)
    ]

    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        result = check_func()
        results.append((name, result))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to start!")
        print("\nNext steps:")
        print("1. Preprocess data : python src/data/preprocess_data.py")
        print("2. Train model     : python src/models/train_arima.py")
        print("3. Launch dashboard: streamlit run dashboard/app.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
