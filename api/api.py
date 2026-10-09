from flask import Flask
from flask import request
from flask import jsonify
import sys
import os
import joblib
regression_model = joblib.load('models/flight_price_model.pkl')  # or regression_model.pkl
classification_model = joblib.load('models/gender_classification_model.pkl')
recommendation_model = joblib.load('models/hotel_recommendation_model.pkl')
# Automatically add the project paths so 'api.utils' is found
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)
sys.path.insert(0, os.path.dirname(this_dir))
sys.path.insert(0, os.path.dirname(os.path.dirname(this_dir)))
from utils import (
    predict_flight_price,
    predict_gender,
    hotel_recommendation
)

app = Flask(__name__)


# Home route
@app.route('/')

def home():

    return 'Voyage Analytics API Running Successfully'


# Flight Price Prediction
@app.route(
    '/predict-flight-price',
    methods=['POST']
)

def flight_prediction():

    try:

        data = request.get_json()

        prediction = predict_flight_price(data)

        return jsonify({

            'Predicted Flight Price': prediction

        })

    except Exception as e:

        return jsonify({

            'Error': str(e)

        })


# Gender Prediction
@app.route(
    '/predict-gender',
    methods=['POST']
)

def gender_prediction():

    try:

        data = request.get_json()

        prediction = predict_gender(data)

        return jsonify({

            'Predicted Gender': prediction

        })

    except Exception as e:

        return jsonify({

            'Error': str(e)

        })


# Hotel Recommendation
@app.route(
    '/recommend-hotels',
    methods=['POST']
)

def recommend_hotels_api():

    try:

        data = request.get_json()

        hotel_name = data['hotel_name']

        recommendations = hotel_recommendation(
            hotel_name
        )

        return jsonify({

            'Recommendations': recommendations

        })

    except Exception as e:

        return jsonify({

            'Error': str(e)

        })


# Run Flask App
if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
