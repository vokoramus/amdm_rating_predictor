from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired

import urllib.request
import json


class ClientDataForm(FlaskForm):
    chords = StringField('Song chords', validators=[DataRequired()])
    view = StringField('view', validators=[DataRequired()])
    star = StringField('star', validators=[DataRequired()])
    date = StringField('date', validators=[DataRequired()])
    url = StringField('url', validators=[DataRequired()])
    text = StringField('text', validators=[DataRequired()])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)


def get_prediction(chords, view, star, date, url, text):
    body = {'chords': chords,
            'view': view,
            'star': star,
            'date': date,
            'url': url,
            'text': text,
            }

    myurl = "http://0.0.0.0:8180/predict"

    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')

    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', str(len(jsondataasbytes)))
    #print (jsondataasbytes)

    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        data['chords'] = request.form.get('chords')
        data['view'] = request.form.get('view')
        data['star'] = request.form.get('star')
        data['date'] = request.form.get('date')
        data['url'] = request.form.get('url')
        data['text'] = request.form.get('text')

        try:
            response = str(get_prediction(
                data['chords'],
                data['view'],
                data['star'],
                data['date'],
                data['url'],
                data['text'],
            ))

            response_dict = json.dumps({"pred": response,
                                   "view_test": int(data['star']) / int(data['view']) * 1000})
            print(response_dict)
        except ConnectionError:
            response_dict = json.dumps({"error": "ConnectionError"})
        # return redirect(url_for('predicted', response=response))
        return redirect(url_for('predicted', response=response_dict))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
