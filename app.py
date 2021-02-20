import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output=["Fraud" if prediction[0]==1 else "Not Fraud"][0]

    return render_template('index.html', prediction_text='The transaction is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output=["Fraud" if prediction[0]==1 else "Not Fraud"]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)