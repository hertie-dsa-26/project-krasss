# _Halcyon_
**Team:** KRASSS

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-web%20app-black?logo=flask)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter)
![uv](https://img.shields.io/badge/uv-dependency%20management-6A5ACD)

> **An interactive Flask application for exploring how weather patterns relate to health outcomes across U.S. counties from 2013 to 2023.**

**Team Members:**

* Santiago Cárdenas Arciniegas
* Sanjeev Dasgupta
* Sophie Latham
* Trayda Murakami
* Konstantina Panagiotopoulou
* Alex Richter
* Rudranshi Vishnoi


## 🌦️ Description

**_Halcyon_** is an interactive Flask-based **data science web application** developed as part of the Data Structures and Algorithms course at the Hertie School.

The project combines public data from NOAA, CDC Places, and the U.S. Census Bureau to examine how **weather patterns** relate to **health outcomes** across U.S. counties between 2013 and 2023. The integrated dataset supports both exploratory analysis and predictive modeling, with a focus on five health outcomes. The project includes a manual implementation of **Kernel Ridge Regression** with **Random Fourier Features approximating the Gaussian kernel** to support transparent modeling for four targets: physical health, mental health, asthma incidence, and stroke incidence. **XGBoost (Extreme Gradient Boosting)** is used to model sleep.

The project also emphasizes **software engineering principles**, including modular design, testing, efficient algorithm implementation, and collaborative development under a Scrum-based workflow.


## 📊 Data

| Dataset          | Unit of Analysis       | Years       | Features                              | Source |
|------------------|----------------------|------------|------------------------------------------|--------|
| NOAA        | U.S. county   | 2013-2023  | Weather data (e.g., temperature, precipitation) | [Link](https://www.ncei.noaa.gov/cdo-web/) |
| CDC Places       | U.S. county          | 2013-2023  | Health outcomes (e.g., asthma, sleep)     | [Link](https://www.cdc.gov/places/tools/data-portal.html) |
| U.S. Census Bureau    | U.S. county           | 2013-2023  | Socioeconomic and demographic variables   | [Link](https://www.census.gov/data/developers/data-sets/acs-5year.html) |

The final dataset contains **6,646 rows** and **41 columns**.


## 🗂️ Repository Structure

```text
project-krasss/
├── app/                                   # Flask web application
│   ├── functions/                         # Model training, preprocessing, evaluation, and scenario logic
│   │   ├── models/                        # Trained model files
│   │   ├── assessment.py
│   │   ├── cross_validator.py
│   │   ├── krr.py
│   │   ├── preprocessing.py
│   │   ├── random_fourier_features.py
│   │   ├── scenarios.py
│   │   ├── splitter.py
│   │   ├── train.py
│   │   └── xgboost_wrapper.py
│   ├── static/
│   ├── templates/                         # HTML templates for the app
│   ├── main.py                            # Main app entry point
│   ├── pyproject.toml
│   ├── style_guide.html
│   └── uv.lock
│
├── data/                                  # App-ready data
│   ├── archive/
│   └── merged_final_transformed.csv
│
├── docs/                                  # Project documentation and app planning files
│   ├── KRASSS_documentation_outline.docx
│   ├── KRASSS_documentation_second_draft.docx
│   ├── app_map.html
│   └── user_requirements.md
│
├── preparation/                           # Data preparation, EDA, and model experimentation
│   ├── data_cleaning/
│   │   ├── data_files/
│   │   └── notebooks/
│   ├── exploratory_data_analysis/
│   └── ml_scripts/
│
├── tests_and_diagnostics/                 # Tests and diagnostic analysis
│   ├── diagnostics.ipynb
│   ├── e2e/
│   ├── integration/
│   └── unit/
│
├── .gitignore
└── README.md
```


## 🚀 Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/hertie-dsa-26/project-krasss.git
   cd project-krasss
   ```

2. Navigate to the Flask app folder:

   ```bash
   cd app
   ```

3. Install `uv` if needed:

   ```bash
   pip install uv
   ```

4. Install the app dependencies:

   ```bash
   uv sync
   ```

5. Run the application:

   ```bash
   uv run python main.py
   ```

6. Once the server starts, open the app in your browser at:

   ```text
   http://127.0.0.1:5000
   ```