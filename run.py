from flask import Flask, request, jsonify
import pandas as pd
from forecast_end_of_month import *

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'UP'})

# @app.route('/forecast', methods=['POST'])
# def forecast():
#     # json = time_series.to_json(orient='split', date_format='iso')
#     json = request.get_json(force=True)
#     time_series = pd.read_json(json, typ='series', orient='split')
#     forcasted_values = forecast_end_of_month_total(time_series, name, params)
#     return jsonify({'forecasted_values': forcasted_values})

if __name__ == '__main__':
    app.run(port=8080, debug=True)


