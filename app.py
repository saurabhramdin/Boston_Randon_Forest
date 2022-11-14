from project_app.utils import BostonHousePrice
from flask import Flask,request,jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to home page"

@app.route('/predict')
def predict():
    input = request.get_json()
    LSTAT = input['LSTAT']
    RM = input['RM']
    DIS = input['DIS']
    CRIM = input['CRIM']
    PTRATIO = input['PTRATIO']
    AGE = input['AGE']
    B = input['B']
    NOX = input['NOX']

    boston = BostonHousePrice(LSTAT,RM,DIS,CRIM,PTRATIO,AGE,B,NOX)
    output = boston.predict_price()

    return jsonify({"Predicted House Price ":output})

if __name__ == '__main__':
    app.run(debug=True)