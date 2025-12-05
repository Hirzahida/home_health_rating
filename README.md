# home_health_rating
Data Science Exercise on Analyzing Publicly Available Data

## Project Introduction
This project focuses on analyzing state?level home care data to identify the key factors that influence star ratings. The analysis began with exploratory data examination using the HH_State_Oct2025.csv file. The modeling code has been refactored into modular functions  and the success criterias are determined based on this list:

1.  Can we develop a reliable model to predict star ratings based on patient improvement measures?
2.  What are the key factors within patient improvement metrics that most strongly influence star ratings.
3.  If flu shot compliance is increased by 10%, what impact would that have on star ratings, and in which states should we prioritize this initiative?

## File Descriptions

The following files are provided to support the analysis:

1.  HH_State_Oct2025.csv – A CSV file containing data on home health quality and associated measurement metrics. The answers to the questions above are based on this dataset.
2.  README.d – A documentation file that includes all necessary information about the projects, and instructions and details required to run the analysis.
3.  HH_rating.ipynb – A Jupyter Notebook file containing the complete analysis workflow described in CRISP-DM, including data preparation and validation, model development, and evaluation. It can be opened in Jupyter Notebook and requires Python 3.3. 
4.  HH_rating.py – An executable Python script containing the same analysis code.

## Quick Start/ Execution

1.  Jupyter Notebook
    - Make sure HH_State_Oct2025.csv is in the same directory as HH_rating.py
    - Ensure all required packages are installed: pandas, numpy, matplotlib, scikit-learn
    - Run the HH_rating.ipynb

2.  Linux/Unix Terminal
    - Create a Python environment with pandas, scikit-learn, and matplotlib
    - Run the refactored script:

        *bash*
        python HH_rating.py

3.  Using a Different Dataset
    - If you intend to run with another CSV dataset, edit the script at the bottom and change the parameters accordingly:
        pythonfilepath="HH_State_Oct2025.csv",
        target_col="Quality of Patient Care Star Rating",
        flu_col="How often the home health team determined whether patients received a flu sh

4.  Expected Outputs
    - Data loading confirmation
    - Basic exploration (columns, data types, missing values, summary stats)
    - Histogram plots for data exploration
    - Model training results (MSE and R²)
    - Top 10 feature importance bar chart
    - What-if analysis results with histogram
    - Top 5 states with biggest improvement

## Results/Discussions

The main findings from the analysis are summarized in the Medium post available [here](https://medium.com/@hirzahida/improving-home-health-care-quality-a-data-driven-approach-a4e345dc19f7?postPublishedType=repub).

## Licensing, Authors, Acknowledgements

This dataset used in this analysis is publicly available [here](	https://data.cms.gov/provider-data/dataset/tee5-ixt5). It was published by Centers for Medicare & Medicaid Services (CMS) on October 22, 2025.  
Portions of the code were adapted from classroom notes and exercises, with some ideas inspired by AI suggestions, then refined to meet the project requirements.
