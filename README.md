# Stock Prediction Project

This project predicts stock prices using machine learning models like LSTM, RNN, Elastic Net, etc. The main dataset is GOOG stock price data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Create Conda Environment](#create-conda-environment)
  - [Activate Conda Environment](#activate-conda-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
- [Git Workflow](#git-workflow)
- [Contact](#contact)

---

## Project Overview

This project uses several machine learning models to predict stock prices based on historical data. Models implemented include:

- LSTM Neural Network
- RNN Neural Network
- Elastic Net Regression
- Other classical regression models

The goal is to analyze and compare model performance on the GOOG stock dataset.

---

## Setup Instructions

### Prerequisites

- Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git installed on your machine
- Python 3.8 or higher (Conda environment will handle this)

---

### Create Conda Environment

To create the Conda environment for this project, run:

```bash
conda create --name stockenv python=3.8 -y
````

This will create a new environment named `stockenv` with Python 3.8.

---

### Activate Conda Environment

Activate the environment using:

```bash
conda activate stockenv
```

---

### Install Dependencies

After activating the environment, install all required packages.

If you have a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Or if you use Conda for dependencies, you can create an `environment.yml` file and run:

```bash
conda env update --file environment.yml --prune
```

**Typical dependencies for this project:**

* numpy
* pandas
* matplotlib
* scikit-learn
* tensorflow
* keras
* flask (if using API)
* jupyterlab (optional)

You can manually install them as well:

```bash
conda install numpy pandas matplotlib scikit-learn -y
pip install tensorflow keras flask
```

---

## Usage

1. Clone the repo:

```bash
git clone https://github.com/your-username/stock-prediction.git
cd stock-prediction
```

2. Activate your environment:

```bash
conda activate stockenv
```

3. Run your training and prediction scripts, for example:

```bash
python train_models.py
```

or launch your Jupyter notebook:

```bash
jupyter notebook GOOG_stock.ipynb
```

---

## Git Workflow

1. Initialize git (only once):

```bash
git init
```

2. Add files:

```bash
git add .
```

3. Commit changes:

```bash
git commit -m "Initial commit"
```

4. Add remote repo:

```bash
git remote add origin https://github.com/your-username/stock-prediction.git
```

5. Push to GitHub:

```bash
git branch -M main
git push -u origin main
```

---

## Contact

For questions or help, please contact:

**Sanket Shinde**
---

*Happy Stock Predicting! 🚀*

