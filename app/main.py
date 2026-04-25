from flask import Flask, render_template, jsonify, request
import pandas as pd
import os

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

# Route: Explore
@app.route('/explore')
def explore():
    years = sorted(df['year'].unique().tolist())
    counties = sorted(df['County name'].unique().tolist())
    climate_types = sorted(df['climate_type_short'].unique().tolist())
    df_columns = df.columns.tolist()

    return render_template(
        'explore.html',
        years=years,
        counties=counties,
        climate_types=climate_types,
        df_columns=df_columns
    )

# Route: Predict
@app.route('/predict')
def predict():
    counties = sorted(df['County name'].unique().tolist())
    return render_template('predict.html', counties=counties)

# MOVE THIS ABOVE app.run()
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

# Replace your existing /api/map-data and /api/snapshot and /api/timeseries routes in main.py with these

@app.route('/api/snapshot')
def snapshot():
    n_states   = df['StateAbbr'].nunique()
    n_counties = df['County name'].nunique()
    n_years    = df['year'].nunique()
    n_vars     = 41
    return jsonify({
        'states':   n_states,
        'counties': n_counties,
        'years':    n_years,
        'n_vars':   n_vars
    })
 


@app.route('/api/map-data')
def map_data():
    health_var = request.args.get('var', 'MHLTH')
    weather_var = request.args.get('weather', '')
    year_start  = int(request.args.get('year_start', 2013))
    year_end    = int(request.args.get('year_end', 2023))

    if health_var not in df.columns:
        return jsonify({"error": "Invalid variable"}), 400

    filtered = df[(df['year'] >= year_start) & (df['year'] <= year_end)]

    agg_dict = {
        'health_val': (health_var, 'mean'),
        'population': ('total_population', 'mean'),
    }
    if weather_var and weather_var in df.columns:
        agg_dict['weather_val'] = (weather_var, 'mean')

    grouped = filtered.groupby(['CountyFIPS', 'County name', 'StateAbbr']).agg(**agg_dict).reset_index()

    health_nat_avg  = round(grouped['health_val'].mean(), 2)
    weather_nat_avg = round(grouped['weather_val'].mean(), 2) if 'weather_val' in grouped.columns else None

    result = []
    for _, row in grouped.iterrows():
        entry = {
            'fips':        str(int(row['CountyFIPS'])).zfill(5),
            'county':      row['County name'].title(),
            'state':       row['StateAbbr'],
            'health_val':  round(row['health_val'], 2),
            'population':  int(row['population']) if not pd.isna(row['population']) else 0,
        }
        if 'weather_val' in grouped.columns:
            entry['weather_val'] = round(row['weather_val'], 2) if not pd.isna(row['weather_val']) else None
        result.append(entry)

    return jsonify({
        'data':             result,
        'health_nat_avg':   health_nat_avg,
        'weather_nat_avg':  weather_nat_avg,
        'health_var':       health_var,
        'weather_var':      weather_var,
        'year_start':       year_start,
        'year_end':         year_end
    })


@app.route('/api/timeseries')
def timeseries():
    health_var  = request.args.get('health', 'MHLTH')
    weather_var = request.args.get('weather', '')
    state       = request.args.get('state', 'all')
 
    result = {}
 
    for var, key in [(health_var, 'health'), (weather_var, 'weather')]:
        if not var or var not in df.columns:
            continue
        if state == 'all':
            grouped = df.groupby('year')[var].mean().reset_index()
        else:
            grouped = df[df['StateAbbr'] == state].groupby('year')[var].mean().reset_index()
 
        national = df.groupby('year')[var].mean().reset_index()
 
        result[key] = {
    'series':   [{'year': int(r['year']), 'value': round(r[var], 2)} for _, r in grouped.iterrows() if not pd.isna(r[var])],
    'national': [{'year': int(r['year']), 'value': round(r[var], 2)} for _, r in national.iterrows() if not pd.isna(r[var])],
    'var':      var
}
 
        # Summary stats
        col = df[var].dropna()
        result[key]['stats'] = {
            'mean':   float(round(col.mean(), 2)),
            'min':    float(round(col.min(), 2)),
            'max':    float(round(col.max(), 2)),
            'range':  float(round(col.max() - col.min(), 2)),
            'median': float(round(col.median(), 2)),
            'std':    float(round(col.std(), 2))
        }
 
    return jsonify({
        'data':  result,
        'state': state
    })
 

# app.run() MUST be the last thing in the file
if __name__ == '__main__':
    app.run(debug=True)
