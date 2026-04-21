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

# app.run() MUST be the last thing in the file
if __name__ == '__main__':
    app.run(debug=True)
