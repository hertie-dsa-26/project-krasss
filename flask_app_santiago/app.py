# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from train import load_model
from scenarios import generate_scenario, SCENARIOS

app = Flask(__name__)

# ── Load data and county list at startup ──────────────────────────────────────
DATA_PATH = "data/merged_final_transformed.csv"
df = pd.read_csv(DATA_PATH)

# Build unique county + state combinations to avoid duplicate county names
# across different states e.g. CLINTON COUNTY exists in MI, IA and NY
county_options = (
    df[["County name", "StateAbbr"]]
    .drop_duplicates()
    .dropna()
    .sort_values(["StateAbbr", "County name"])
)
COUNTIES = [
    {
        "label":  f"{row['County name']} — {row['StateAbbr']}",
        "county": row["County name"],
        "state":  row["StateAbbr"]
    }
    for _, row in county_options.iterrows()
]

TARGETS = ['BPHIGH', 'CASTHMA', 'COPD', 'MHLTH', 'PHLTH', 'STROKE', 'SLEEP']

print(f" Data loaded: {len(df)} rows")
print(f" County options: {len(COUNTIES)}")
print(f" Targets: {TARGETS}")
print(f" Scenarios: {list(SCENARIOS.keys())}")


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        counties=COUNTIES,
        targets=TARGETS,
        scenarios=SCENARIOS
    )


@app.route("/predict", methods=["POST"])
def predict():
    # ── Get form inputs ───────────────────────────────────────────
    county_state = request.form["county_state"]
    county, state = county_state.split("|")
    target = request.form["target"]
    scenario_key = request.form["scenario"]

    print(
        f"  → county={county} | state={state} | target={target} | scenario={scenario_key}")

    # ── Load pre-fitted model and preprocessor ────────────────────
    krr, preprocessor = load_model(target)

    # ── Generate synthetic future rows ────────────────────────────
    X_future, future_years = generate_scenario(df, county, state, scenario_key)

    # ── Preprocess and predict ────────────────────────────────────
    X_scaled = preprocessor.transform(X_future.to_numpy())
    y_pred = krr.predict(X_scaled)

    # ── Build results ─────────────────────────────────────────────
    results = list(zip(future_years, y_pred.round(4).tolist()))

    return render_template(
        "index.html",
        counties=COUNTIES,
        targets=TARGETS,
        scenarios=SCENARIOS,
        results=results,
        county=county,
        state=state,
        target=target,
        scenario_key=scenario_key,
        description=SCENARIOS[scenario_key]["description"]
    )


if __name__ == "__main__":
    app.run(debug=True)
