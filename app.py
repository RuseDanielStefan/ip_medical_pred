from flask import Flask, request
import json
import mysql.connector
import predict
from datetime import date
import predict


app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello from flask'

id = 2


@app.route('/index', methods=['GET', 'POST'])
def index():
    data_dic = json.loads(request.data, encoding='UTF-8')
    cnx = mysql.connector.connect(
        user='b380f338c76a8d', password='8768bb5c', host='eu-cdbr-west-02.cleardb.net', database='heroku_c4a6a99da4e3951')
    global id
    id += 1 
    patient_id = 1
    cursor = cnx.cursor()
    calories = 1000
    pulse = data_dic['pulse']
    # steps = data_dic['steps']
    temperature = data_dic['temperature']
    today = data_dic['today']
    water = data_dic['water']
    weight = data_dic['weight']
    sql = ("INSERT INTO daily_data"
           " (id, patient_id, water, weight, pulse, temperature, calories, day)"
           " values (%s, %s, %s, %s, %s, %s, %s, %s)")
    day, month, year = today.split('-')
    day, month, year = int(day), int(month), int(year)
    today = date(year, month, day)

    daily_data = (id, patient_id, water, weight, pulse,
                  temperature, calories, today)
    cursor.execute(sql, daily_data)
    cnx.commit()

    print(data_dic)
    return json.dumps(data_dic)


@app.route('/predict_hd', methods=['POST'])
def get_result():
    data = request.get_json()
    return predict.classify(data, k=30)


@app.route('/hello')
def salut():
    return 'hello from hello'


if __name__ == '__main__':
    app.run()
