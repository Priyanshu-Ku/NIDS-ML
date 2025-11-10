# 🚀 Quick Start Guide - NIDS-ML

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] pip package manager
- [ ] Administrator/Root access (for packet capture)
- [ ] At least 8GB RAM
- [ ] 10GB free disk space

---

## Step-by-Step Setup

### 1️⃣ Environment Setup

```powershell
# Navigate to project directory
cd "c:\Users\Priyanshu\Desktop\Main\ML\Network Intrusion Detection using ML (NIDS)\NIDS-ML"

# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Upgrade pip
python -m pip install --upgrade pip
```

### 2️⃣ Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, sklearn, scapy, streamlit; print('✅ All packages installed!')"
```

### 3️⃣ Download Dataset

**Option A: CICIDS 2017 (Recommended)**

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download all CSV files (Monday through Friday)
3. Create folder: `data/raw/CICIDS2017/`
4. Place all CSV files in this folder

**Option B: NSL-KDD (Fallback)**

1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download KDDTrain+.txt and KDDTest+.txt
3. Create folder: `data/raw/NSL-KDD/`
4. Place files in this folder

### 4️⃣ Project Workflow

```powershell
# Navigate to src directory
cd src

# Step 1: Preprocess Data
python data_preprocessing.py

# Step 2: Feature Selection
python feature_selection.py

# Step 3: Train Models
python model_training.py

# Step 4: Generate SHAP Explanations
python shap_explainability.py

# Step 5: Launch Dashboard
streamlit run dashboard.py
```

### 5️⃣ Real-Time Detection (Optional)

```powershell
# Run as Administrator
# Right-click PowerShell -> "Run as Administrator"

cd src
python realtime_detection.py
```

---

## 📝 Common Issues & Solutions

### Issue 1: Scapy Installation Failed

**Solution:**

```powershell
pip install scapy --user
# Or install Npcap: https://npcap.com/#download
```

### Issue 2: TensorFlow Import Error

**Solution:**

```powershell
# For CPU-only version
pip install tensorflow-cpu

# Or use PyTorch instead
pip install torch torchvision
```

### Issue 3: Permission Denied (Packet Capture)

**Solution:**

- Windows: Run terminal as Administrator
- Linux/Mac: Use `sudo python realtime_detection.py`

### Issue 4: Memory Error

**Solution:**

- Process dataset in chunks
- Reduce SMOTE sampling
- Use smaller feature set

---

## 🎯 Verification Tests

### Test 1: Check Project Structure

```powershell
python -c "import os; print('\n'.join([d for d in ['data', 'src', 'models', 'logs'] if os.path.exists(d)]))"
```

### Test 2: Test Imports

```powershell
python -c "from src.utils import setup_logging; from src.config import BASE_DIR; print('✅ Modules OK')"
```

### Test 3: Check Dataset

```powershell
python -c "import pandas as pd; import os; print('✅ Dataset found' if os.path.exists('data/raw/dataset.csv') else '❌ Dataset missing')"
```

---

## 📊 Expected Timeline

| Phase     | Task                       | Time Estimate |
| --------- | -------------------------- | ------------- |
| Setup     | Environment + Dependencies | 30 mins       |
| Data      | Download + Preprocessing   | 1-2 hours     |
| Features  | Selection + Engineering    | 30 mins       |
| Training  | All Models                 | 2-4 hours     |
| SHAP      | Explainability             | 30 mins       |
| Dashboard | Setup + Testing            | 30 mins       |
| **Total** |                            | **5-8 hours** |

---

## 🎓 Learning Path

### For Beginners:

1. Start with `data_preprocessing.py` - understand data cleaning
2. Explore `feature_selection.py` - learn feature engineering
3. Run `model_training.py` - see ML in action
4. Use `dashboard.py` - visualize results

### For Advanced Users:

1. Modify hyperparameter grids in `config.py`
2. Implement custom feature extraction in `realtime_detection.py`
3. Add new models in `model_training.py`
4. Enhance SHAP visualizations in `shap_explainability.py`

---

## 📚 Next Steps

After completing this setup, proceed to:

1. **Part 2**: Data Preprocessing Implementation
2. **Part 3**: Model Training & Evaluation
3. **Part 4**: Real-time Detection System
4. **Part 5**: Dashboard Deployment

---

## 🆘 Getting Help

- Check `logs/` directory for error messages
- Review module docstrings for usage examples
- See `README.md` for detailed documentation
- Open GitHub issues for bugs

---

## ✅ Final Checklist

- [ ] Virtual environment activated
- [ ] All packages installed without errors
- [ ] Dataset downloaded and placed in `data/raw/`
- [ ] Project structure verified
- [ ] Test imports successful
- [ ] Ready to start preprocessing!

**🎉 You're all set! Let's build an amazing NIDS system!**
