
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
    prediction = model.predict_proba(final_features)[:,1][0]

    if prediction>=0.35 and  prediction<=0.49:
        output="This customer needs motivation"
    elif prediction>=0.50:
        output="This customer has real interest in the product" 
    else:
        output="This customer is just looking"

    return render_template('index.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
