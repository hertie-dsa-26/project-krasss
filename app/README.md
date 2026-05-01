# To read for Sanjeev 

## Quick note on file size

So the models can get a bit big with pickle. I am not entirely sure how it is operating but it could be that pickle is concatenating. I have a suspiscion that this is the case. I ran it twice and the second time I tried to push everything I was getting errors like 
```bash 
remote: Resolving deltas: 100% (13/13), completed with 8 local objects.
remote: error: Trace: db095de2bad31cd2f9cb6dd902f20e06195c86704f5785a125e1b4ce53bb74b0
remote: error: See https://gh.io/lfs for more information.
remote: error: File app/functions/models/MHLTH.pkl is 229.61 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File app/functions/models/PHLTH.pkl is 229.61 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File app/functions/models/CASTHMA.pkl is 229.61 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File app/functions/models/STROKE.pkl is 229.61 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
```

As a result, I think that if we run it once thats fine. But also potentially adding a line about deleting before creating a model. I didn't do this because I wasn't sure how RFF and XGBoost would be implemented. I imagine we would be creating a bunch of different pkl files specific for the type of model and the variable? But this is a consideration with understanding pickle files. I included the line below because I did have to increase my buffersize after running the models only once! 

**If you run train.py you might have an issue I had with the upload size, I used the following line for a quick fix:**
`git config --global http.postBuffer 1048576000` [Source](https://stackoverflow.com/questions/6842687/the-remote-end-hung-up-unexpectedly-while-git-cloning)

I will paste this text into the pull request but I wanted to include a few words. The next steps to me are pretty straightforward. We want to implement RFF. I put  a comment where I think that is best suited in train.py it is in the tune and evaluate section. I didn't implement that entirely because I wanted you to have a look at the content and I am less confident on my understanding of RFF. 

The app looks nice! I think Santiago did some updating of the index so when you run it, it appears nicely. I ran the train.py twice once before fixing the structure of app.py and its dependencies and again afterwards to ensure that it still produced models. But I removed them from the commit due to the note on file size above. However the script ran with no error messages and no .pkl files created. 

There needs to be a change in the scenarios. The predictions are likely accurate but given its just temperature by a small increase the values barely change for some counties. For other counties, there is literally no change in the values at all. **Fixing scenarios ie updating it based on what is _interesting_ to show prediction wise is probably the thing that requires the most attention.** I integrated your comment about baseline year being hardcoded, I agree it shouldn't be. 

The models are trained fine with KRR, RFF, and imported XGBoost it should be even better. **The pyproject.toml will need to be updated for XGBoost!**

I decided against adding run_grid_search as it is done inside train.py under tune and evaluate but it just doesn't print the scores throughout (like what we do in lab). We can add it but I don't think it is necessary to do so. 

The file that is doing the bulk of the work and will need to be updated for RFF and XGBoost is train.py. 

# Understanding the code for the Krass App

Folder Structure
```text
app/
├── app.py                  ← Flask demo app — entry point
├── main.py 
├── pyproject.toml      <- used for initializing the dependencies 
├── templates/
│   ├── predict.html
│   └── index.html          ← minimal HTML form for the demo
├── functions/                
│    ├── train.py                ← trains and saves all models — run once!! important
│    ├── scenarios.py            ← generates synthetic future data for predictions
│    ├── splitter.py             ← data preparation and time series splitting
│    ├── preprocessing.py        ← encoding, imputation and scaling
│    ├── krr.py                  ← Kernel Ridge Regression model
│    ├── assessment.py           ← MSE and R² scoring
│    └── cross_validator.py      ← cross validation loop

```
# How to run the app
The app uses its own venv to prevent any error.
```bash
uv sync
uv run python app.py
```
To retrain all models from scratch:

`uv run python train.py`

This will take a while. It runs a full grid search for each health variable.

# Modulal descriptions 


- splitter.py contains prepare_data() and the Splitter class. prepare_data() loads the dataframe, drops the non-target health variables and identifier columns, and returns X, y and years. Splitter takes those arrays and produces time series folds via time_series_splits() and the final train/test split via get_test_split(). This is the first thing called in the pipeline.
- preprocessing.py contains detect_categorical_columns(), SimpleImputer, StandardScaler, OneHotEncoder and Preprocessor. These are independent classes connected through composition inside Preprocessor,  not through inheritance. Preprocessor is the only class called from outside this file. It applies encoding → imputation → scaling in that order. The fit/transform separation is strictly enforced — fit_transform() is only ever called on training data.
- krr.py  contains gaussian_kernel() and KernelRidgeRegression. 
- assessment.py contains Assessment with two methods  mean_squared_error() and r2_score(). This class was created by Sanjeev as well and it  computes scores and does not manage any CV loop or state. It is used only for final test set evaluation.
- cross_validator.py contains CrossValidator which owns the full CV loop. It imports Preprocessor directly and creates a fresh instance per fold to prevent data leakage between folds. It calls Assessment internally for scoring and returns a dictionary with mse and r2 lists. start_fold controls which folds count toward hyperparameter selection.
- train.py orchestrates the full pipeline for each health variable. It calls splitter.py, cross_validator.py, preprocessing.py, krr.py and assessment.py in sequence. After fitting the final model it saves both the KRR and the preprocessor together as a .pkl file via save_model(). It also contains load_model() which is imported by app.py.
- scenarios.py **TO BE UPDATED!!!** scontains the SCENARIOS dictionary and generate_scenario(). It takes the 2023 row for a selected county as a baseline, creates 10 synthetic (hypothetical for now) future rows and applies the corresponding climate changes. It returns a raw unscaled feature matrix that must go through preprocessor.transform() before prediction. County names are not unique across states so it uses state_abbr to uniquely identify a county (e.g. CLINTON COUNTY exists in Michigan, Iowa and New York). The app was initially crashing because I hadn't considered the possibility of duplicate names. Including the state fixes it, but we could also consider using the FIPS code if necessary to properly identify counties.
- randon_fourier_features.py 
- app.py is a minimal Flask app with two routes. GET / renders the form. POST /predict loads the model, generates scenario data, preprocesses and predicts. The full prediction flow is:

```text
load_model(target)              → krr, preprocessor  from train.py
generate_scenario(...)          → X_future, years     from scenarios.py
preprocessor.transform(X)      → X_scaled            from preprocessing.py
krr.predict(X_scaled)          → y_pred              from krr.py
```