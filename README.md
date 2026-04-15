# _[Insert Project (App) Name]_
Team: **KRASSS**

## Team Members:

* Santiago Cárdenas Arciniegas
* Sanjeev Dasgupta
* Sophie Latham
* Trayda Murakami
* Konstantina Panagiotopoulou
* Alex Richter
* Rudranshi Vishnoi


## Description

_[Insert Project (App) Name]_ is an interactive **data science web application** developed as part of the Data Structures and Algorithms (DSA) course at the Hertie School.

The project integrates multiple public datasets to explore the relationship between **weather patterns** and **health outcomes** (physical and mental distress, asthma, sleep) across U.S. cities and counties. By combining data from NOAA, the CDC, and the U.S. Census, it constructs a unified dataset that spans years 2013--2022 and supports both exploratory analysis and predictive modeling.

Beyond **data science functionality**, the project emphasizes **software engineering principles**, including modular system design, testing, efficient algorithm implementation, and collaborative development under a Scrum-based workflow. It aims to elucidate the relationship between weather and health while offering a transparent and reproducible implementation of core machine learning techniques **without relying on external ML libraries** for the core model.


## Features

* **Interactive exploration** of weather, health, and socioeconomic data.  
* **Dynamic visualizations** that respond to user input.  
* **Integrated dataset** spanning multiple public sources (NOAA, CDC, and U.S. Census).  
* Implementation of **Kernel Ridge Regression from scratch**.  
* **Flask-based web application** for end-to-end analysis.  


## Data
The project integrates multiple publicly available datasets, aligned at the **Metropolitan Statistical Area (MSA) level** over the period **2013–2022**:

| Dataset          | Unit of Analysis       | Years       | Description                              | Source |
|------------------|----------------------|------------|------------------------------------------|--------|
| NOAA GSOY        | Station → MSA-year   | 2013--2022  | Weather data (e.g., temperature, precipitation) | [NOAA](https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-year) |
| CDC Places       | Place / MSA          | 2013--2022  | Health outcomes (e.g., asthma, sleep)     | [CDC](https://data.cdc.gov/browse?category=500+Cities+%26+Places) |
| US Census ACS    | MSA (CBSA)           | 2013--2022  | Socioeconomic and demographic variables   | [U.S. Census](https://data.census.gov/table) |

The final dataset contains **6,646 rows** and **52 columns**.


## Methodology
The project follows an end-to-end analytical pipeline:

1. **Data Collection and Integration**: Data is collected from NOAA, CDC Places, and the U.S. Census and aligned at the MSA level.
2. **Data Cleaning and Preprocessing**: _[fill in]_
3. **Exploratory Data Analysis:** Statistical summaries and visualizations are used to identify patterns and relationships within and between variables.
4. **Feature Engineering:** _[fill in]_
5. **Model Implementation:** A Kernel Ridge Regression is implemented from scratch, without assistance from external Machine Learning Libraries.
6. **Visualization and User Interaction:** Results are presented through an interactive web application that uses Flask.


## Development Workflow
Throughout the process, [standard developing practices](https://dev.to/speaklouder/be-a-better-developer-with-these-git-good-practices-13j9) were followed, including:

* Meaningful commit messages.
* Pull requests and code reviews.
* Regular pulls from main.
* Use of branches for new features and experimentation.
* Regular synchronization with main.


## Repository Structure

```text
project-krasss/
├── data/                        # fill in
├── data_cleaning/               # fill in
├── exploratory_data_analysis/   # EDA notebooks
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
└── README.md
```


## Installation & Usage
_[Fill in instructions on how to clone the repository, install dependencies, and run the Flask app.]_


## Deadline
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
