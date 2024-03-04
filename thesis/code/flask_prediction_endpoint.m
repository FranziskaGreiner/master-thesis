app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_type = data.get('model_type', 'tft')  # default: tft
    duration = data['duration']  # in minutes
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d %H:%M:%S')  # format: 'YYYY-MM-DD HH:MM:SS'
    country = data['country']
