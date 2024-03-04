from flask import Flask, render_template, request
import joblib

model = joblib.load('model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Model', methods=['post'])
def model_pred():
    feat_59 = float(request.form.get('features_59'))
    feat_76 = float(request.form.get('features_76'))
    feat_79 = float(request.form.get('features_79'))
    feat_95 = float(request.form.get('features_95'))
    feat_100 = float(request.form.get('features_100'))
    feat_114 = float(request.form.get('features_114'))
    feat_129 = float(request.form.get('features_129'))
    feat_159 = float(request.form.get('features_159'))
    feat_160 = float(request.form.get('features_160'))
    feat_210 = float(request.form.get('features_210'))
    feat_468 = float(request.form.get('features_468'))
    feat_511 = float(request.form.get('features_511'))
    feat_589 = float(request.form.get('features_589'))

    result = model.predict([[feat_59, feat_76, feat_79, feat_95, feat_100, feat_114, feat_129,
                            feat_159, feat_160, feat_210, feat_468, feat_511, feat_589]])
    if result[0] == 1:
        return 'Semiconductor is not defective'
    else:
        return 'Semiconductor is defective'


# app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0')

