from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions"))
from functions import train
from functions import scenarios

app = Flask(__name__)

# Load dataset once when app starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "merged_final_transformed.csv")

df = pd.read_csv(CSV_PATH)

# Route: Home
@app.route("/")
def home():
    return render_template("index.html")

# Route: Documentation
@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/explore')
def explore():
    years = sorted(df['year'].unique().tolist())
    counties = sorted(df['County name'].unique().tolist())
    climate_types = sorted(df['climate_type_short'].unique().tolist())
    df_columns = df.columns.tolist()

    state_names = {
        'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California',
        'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
        'HI':'Hawaii','IA':'Iowa','ID':'Idaho','IL':'Illinois','IN':'Indiana',
        'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','MA':'Massachusetts','MD':'Maryland',
        'ME':'Maine','MI':'Michigan','MN':'Minnesota','MO':'Missouri','MS':'Mississippi',
        'MT':'Montana','NC':'North Carolina','ND':'North Dakota','NE':'Nebraska',
        'NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NV':'Nevada',
        'NY':'New York','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania',
        'RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee',
        'TX':'Texas','UT':'Utah','VA':'Virginia','VT':'Vermont','WA':'Washington',
        'WI':'Wisconsin','WV':'West Virginia','WY':'Wyoming'
    }
    states = [(abbr, state_names.get(abbr, abbr)) 
              for abbr in sorted(df['StateAbbr'].unique().tolist())
              if abbr in state_names]

    return render_template(
        'explore.html',
        years=years,
        counties=counties,
        climate_types=climate_types,
        df_columns=df_columns,
        states=states
    )

# Route: Predict
@app.route("/predict", methods=["GET", "POST"])
def predict():
    county_options = (
        df[["County name", "StateAbbr"]]
        .drop_duplicates()
        .dropna()
        .sort_values(["StateAbbr", "County name"])
    )
    COUNTIES = [
        {
            "label":  f"{row['County name']}, {row['StateAbbr']}",
            "county": row["County name"],
            "state":  row["StateAbbr"]
        }
        for _, row in county_options.iterrows()
    ]

    TARGETS = ['CASTHMA', 'MHLTH', 'PHLTH', 'STROKE', 'SLEEP']

    print(f" Data loaded: {len(df)} rows")
    print(f" County options: {len(COUNTIES)}")
    print(f" Targets: {TARGETS}")
    print(f" Scenarios: {list(scenarios.SCENARIOS.keys())}")

    error = None
    results = None
    historical = []
    county = state = target = scenario_key = None

    if request.method == "POST":
        try:
            county_state = request.form["county_state"]
            county, state = county_state.split("|")
            target = request.form["target"]
            scenario_key = request.form["scenario"]

            print(
                f"  → county={county} | state={state} | target={target} | scenario={scenario_key}")

            model, preprocessor = train.load_model(target)

            X_future, future_years = scenarios.generate_scenario(
                df, county, state, scenario_key)

            X_scaled = preprocessor.transform(X_future.to_numpy())
            y_pred, lower, upper = model.predict_interval(X_scaled)

            results = list(zip(
                future_years,
                y_pred.round(2).tolist(),
                lower.round(2).tolist(),   # add lower bound
                upper.round(2).tolist()    # add upper bound
            ))

            hist = (
                df[(df["County name"] == county) & (df["StateAbbr"] == state)]
                .groupby("year")[target]
                .mean()
                .dropna()
                .sort_index()
            )
            historical = [{"year": int(y), "value": round(float(v), 2)} for y, v in hist.items()]

        except Exception as e:
            error = "There was a prediction error. Please try again."
            historical = []
            print(f"Error during prediction: {e}")

    return render_template(
        "predict.html",
        counties=COUNTIES,
        targets=TARGETS,
        scenarios=scenarios.SCENARIOS,
        results=results,
        error=error,
        historical=historical if not error else [],
        county=county if not error else None,
        state=state if not error else None,
        target=target if not error else None,
        scenario_key=scenario_key if not error else None,
        description=scenarios.SCENARIOS[scenario_key]["description"] if (
            scenario_key and not error) else None
    )

@app.route('/api/summary', methods=['POST'])
def summary_stats():
    data = request.json
    column = data.get("column")

    if column not in df.columns:
        return jsonify({"error": "Invalid column name"}), 400

    stats = df[column].describe().to_dict()

    return jsonify({
        "column": column,
        "stats": stats
    })


@app.route('/api/snapshot')
def snapshot():
    n_states   = df['StateAbbr'].nunique()
    n_counties = df['County name'].nunique()
    n_years    = df['year'].nunique()
    n_vars     = 39
    return jsonify({
        'states':   n_states,
        'counties': n_counties,
        'years':    n_years,
        'n_vars':   n_vars
    })
 


@app.route('/api/map-data')
def map_data():
    health_var  = request.args.get('var', 'MHLTH')
    weather_var = request.args.get('weather', '')
    demo_vars   = request.args.get('demo', '')  # comma-separated list e.g. "median_household_income,pct_less_than_hs"
    year_start  = int(request.args.get('year_start', 2013))
    year_end    = int(request.args.get('year_end', 2023))
 
    if health_var not in df.columns:
        return jsonify({"error": "Invalid variable"}), 400
 
    filtered = df[(df['year'] >= year_start) & (df['year'] <= year_end)]
 
    # Build aggregation dict
    agg_dict = {
        'health_val': (health_var, 'mean'),
        'population': ('total_population', 'mean'),
    }
    if weather_var and weather_var in df.columns:
        agg_dict['weather_val'] = (weather_var, 'mean')
 
    # Add demographic variables
    demo_list = [d for d in demo_vars.split(',') if d and d in df.columns] if demo_vars else []
    for i, dv in enumerate(demo_list[:3]):  # max 3
        agg_dict[f'demo_{i}'] = (dv, 'mean')
 
    grouped = filtered.groupby(['CountyFIPS', 'County name', 'StateAbbr']).agg(**agg_dict).reset_index()
 
    health_nat_avg  = round(grouped['health_val'].mean(), 2)
    weather_nat_avg = round(grouped['weather_val'].mean(), 2) if 'weather_val' in grouped.columns else None
 
    result = []
    for _, row in grouped.iterrows():
        entry = {
            'fips':       str(int(row['CountyFIPS'])).zfill(5),
            'county':     row['County name'].title(),
            'state':      row['StateAbbr'],
            'health_val': round(row['health_val'], 2),
            'population': int(row['population']) if not pd.isna(row['population']) else 0,
        }
        if 'weather_val' in grouped.columns:
            entry['weather_val'] = round(row['weather_val'], 2) if not pd.isna(row['weather_val']) else None
        # Add demo values
        for i, dv in enumerate(demo_list[:3]):
            val = row.get(f'demo_{i}')
            entry[f'demo_{i}'] = round(val, 1) if val is not None and not pd.isna(val) else None
        result.append(entry)
 
    return jsonify({
        'data':            result,
        'health_nat_avg':  health_nat_avg,
        'weather_nat_avg': weather_nat_avg,
        'health_var':      health_var,
        'weather_var':     weather_var,
        'demo_vars':       demo_list,
        'year_start':      year_start,
        'year_end':        year_end
    })
 


@app.route('/api/timeseries')
def timeseries():
    health_var  = request.args.get('health', 'MHLTH')
    weather_var = request.args.get('weather', '')
    state       = request.args.get('state', 'all')
 
    result = {}

    def weighted_mean_by_year(data, var):
        """Population-weighted mean per year."""
        def wmean(g):
            w = g['total_population'].fillna(0)
            v = g[var]
            mask = v.notna() & (w > 0)
            if mask.sum() == 0:
                return np.nan
            return float(np.average(v[mask], weights=w[mask]))
        series = data.groupby('year').apply(wmean).reset_index()
        series.columns = ['year', var]
        return series

    def weighted_stats(data, var):
        """Population-weighted summary stats."""
        w = data['total_population'].fillna(0)
        v = data[var]
        mask = v.notna() & (w > 0)
        v, w = v[mask], w[mask]
        if len(v) == 0:
            return {}
        wmean = float(np.average(v, weights=w))
        # weighted std
        wstd  = float(np.sqrt(np.average((v - wmean)**2, weights=w)))
        return {
            'mean':   round(wmean, 2),
            'min':    float(round(v.min(), 2)),
            'max':    float(round(v.max(), 2)),
            'range':  float(round(v.max() - v.min(), 2)),
            'median': float(round(v.median(), 2)),
            'std':    round(wstd, 2)
        }

    for var, key in [(health_var, 'health'), (weather_var, 'weather')]:
        if not var or var not in df.columns:
            continue

        if state == 'all':
            subset = df
        else:
            subset = df[df['StateAbbr'] == state]

        grouped  = weighted_mean_by_year(subset, var)
        national = weighted_mean_by_year(df, var)

        result[key] = {
            'series':   [{'year': int(r['year']), 'value': round(r[var], 2)} for _, r in grouped.iterrows() if not pd.isna(r[var])],
            'national': [{'year': int(r['year']), 'value': round(r[var], 2)} for _, r in national.iterrows() if not pd.isna(r[var])],
            'var':      var
        }

        # National weighted stats
        result[key]['stats'] = weighted_stats(df, var)

        # State-specific weighted stats
        if state != 'all':
            state_data = df[df['StateAbbr'] == state]
            if len(state_data) > 0:
                result[key]['state_stats'] = weighted_stats(state_data, var)

    return jsonify({
        'data':  result,
        'state': state
    })

@app.route('/api/timeseries/heatmap')
def timeseries_heatmap():
    var    = request.args.get('var', 'MHLTH')
    states = request.args.get('states', '')
    
    if var not in df.columns:
        return jsonify({"error": "Invalid variable"}), 400
    
    state_list = states.split(',') if states else sorted(df['StateAbbr'].unique().tolist())
    
    result = {}
    for state in state_list:
        grouped = df[df['StateAbbr'] == state].groupby('year')[var].mean().reset_index()
        result[state] = {
            int(r['year']): round(r[var], 2) 
            for _, r in grouped.iterrows() 
            if not pd.isna(r[var])
        }
    
    return jsonify({'data': result, 'variable': var})
 

# app.run() MUST be the last thing in the file
if __name__ == '__main__':
    app.run(debug=True)