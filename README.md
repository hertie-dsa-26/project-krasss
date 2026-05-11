# _Halcyon_
**Team:** KRASSS

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-web%20app-black?logo=flask)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter)
![uv](https://img.shields.io/badge/uv-dependency%20management-6A5ACD)

> **An interactive Flask application for exploring how weather patterns relate to health outcomes across U.S. metropolitan areas from 2013 to 2023.**

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

The project combines public data from NOAA, the CDC, and the U.S. Census to examine how **weather patterns** relate to **health outcomes** across U.S. metropolitan areas between 2013 and 2023. The integrated dataset supports both exploratory analysis and predictive modeling, with a focus on the following outcomes: physical distress, mental distress, asthma, and sleep.

The project includes a manual implementation of **Kernel Ridge Regression** (with a Gaussian kernel) to support transparent and interpretable modeling. It also emphasizes **software engineering principles**, including modular design, testing, efficient algorithm implementation, and collaborative development under a Scrum-based workflow.


## ✨ Features

* **Interactive exploration and visualization** of weather, health, and socioeconomic data.  
* An **integrated dataset** spanning multiple public sources (NOAA, CDC, and U.S. Census).  
* A custom implementation of **Kernel Ridge Regression** with a Gaussian kernel.  
* A **Flask-based interface** for end-to-end analysis.  


## 📊 Data

| Dataset          | Unit of Analysis       | Years       | Features                              | Source |
|------------------|----------------------|------------|------------------------------------------|--------|
| NOAA GSOY        | U.S. county   | 2013-2023  | Weather data (e.g., temperature, precipitation) | [NOAA](https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-year) |
| CDC Places       | U.S. county          | 2013-2023  | Health outcomes (e.g., asthma, sleep)     | [CDC](https://data.cdc.gov/browse?category=500+Cities+%26+Places) |
| US Census ACS    | U.S. county           | 2013-2023  | Socioeconomic and demographic variables   | [U.S. Census](https://data.census.gov/table) |

The final dataset contains **6,646 rows** and **41 columns**.


## 🗂️ Repository Structure

```text
project-krasss/
├── data/                        
├── data_cleaning/               
├── exploratory_data_analysis/   
│   ├── eda_univariate.ipynb
│   └── eda_bivariate.ipynb
├── ml_scripts/                  # Model experimentation on health outcomes
│   ├── casthma_ml_algo.ipynb
│   ├── mhlth_ml_algo.ipynb
│   ├── phlth_ml_algo.ipynb
│   ├── sleep_ml_algo.ipynb
│   ├── testing_ml_algorithms.ipynb
│   └── implementation_guidance.ipynb
├── app/                         # Flask web application
├── docs/                        # Project documentation, user requirements, and app map
├── .gitignore
└── README.md
```


## 🚀 Getting Started
1. Clone the repository with:

   ```bash
   git clone https://github.com/hertie-dsa-26/project-krasss.git
   cd project-krasss
   ```

2. Navigate to the `app/` folder:

   ```bash
   cd app
   ```

3. Install the dependencies using `uv`:

    ```bash
    uv sync
    ```

    If `uv` is not installed, you can install it first:

    ```bash
    pip install uv
    ```

4. Run the application:

   ```bash
   uv run python main.py
   ```

5. Once the server starts, open the app in your browser at:

   ```text
   http://127.0.0.1:5000
   ```


_Deadline:_
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
