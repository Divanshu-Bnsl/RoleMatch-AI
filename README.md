# RoleMatch AI

RoleMatch AI is a resume analysis project that helps evaluate resumes against job-role requirements using machine learning.

## Project Structure

- `app.py` - main application entry point
- `train.py` - model training pipeline
- `build_skill_db.py` - skill database preparation
- `plot_roc.py` - ROC curve generation script
- `requirements.txt` - Python dependencies
- `data/resume.csv` - input dataset
- `model/` - trained model artifacts

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Train the model:

```bash
python train.py
```

Run the app:

```bash
python app.py
```
