import os
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import extractFeatures

project_root = os.path.dirname(os.path.realpath('__file__'))
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
app = Flask(__name__, template_folder=template_path, static_folder=static_path)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/enpred')
def index1():
    return render_template('home.html')


@app.route('/cite')
def cite():
    return render_template('cite.html', title='Citation')


@app.route('/read')
def read():
    return render_template('readme.html', title='ReadMe')


@app.route('/supl')
def supl():
    return render_template('supl.html', title='Supplementary Data')


@app.route('/about')
def about():
    return render_template('about.html', title='About us')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


class PredForm(Form):
    sequence = TextAreaField(u'Enter the DNA query sequence(s) in FASTA format', [validators.DataRequired()])


def SimpleParser(sequence):
    seq = sequence.split('\n')
    re = ''
    for x in seq:
        re = re + x[:len(x) - 2]
    return re


def SimpleFastaParser(fasta_sequence):
    seq = fasta_sequence.split('\n')
    seq = seq[1:]
    re = ''
    for x in seq:
        re = re + x[:len(x) - 2]
    return re


@app.route("/pred", methods=['GET', 'POST'])
def pred():
    form = PredForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        input_seq = request.form['sequence']
        results = []
        if '>' in input_seq:
            sequences = []
            for ss in input_seq.split('>'):
                sequence = SimpleFastaParser(ss)
                sequences.append(sequence)
        else:
            sequences = []
            for ss in input_seq.split('>'):
                sequence = SimpleParser(ss)
                sequences.append(sequence)
        for ss in sequences:
            if ss != '':
                result = extractFeatures.extractFeatures.feature_result(ss)
                results.append(result)

        return resultPage(results)
    return render_template('pred.html', form=form, title="Prediction")


def resultPage(result):
    return render_template('result.html', result=result, title="Results")


if __name__ == "__main__":
    app.run()
