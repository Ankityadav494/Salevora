# ⚡ Quick Start Guide

Get your Sales Forecasting project up and running in 5 minutes!

## 🚀 Setup (5 minutes)

### 1. Install Dependencies
```bash
# Activate virtual environment first (if created)
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install all packages
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python src/data/generate_sample_data.py
```

This creates `data/raw/sales_data.csv` with realistic sales data.

### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 What You'll See

1. **Overview Page**: Key metrics and sales trends
2. **Time Series Analysis**: Decomposition and patterns
3. **Forecasting**: Generate predictions (after training models)
4. **Data Explorer**: Filter and explore your data

## 🎯 Next Steps

1. **Explore Data**: Open `notebooks/01_data_exploration.ipynb` in Jupyter
2. **Train Models**: Run `python src/training/train.py`
3. **Test API**: Start API with `python api/app.py`

## 💡 Tips

- Start with the dashboard to see your data
- Use Jupyter notebooks for exploration
- Train models before using forecasting features
- Check `PROJECT_GUIDE.md` for detailed steps

## 🆘 Need Help?

- Check `README.md` for full documentation
- See `PROJECT_GUIDE.md` for step-by-step instructions
- Verify all files are in place: `data/raw/sales_data.csv` should exist

---

**Ready to build something amazing! 🎓**
