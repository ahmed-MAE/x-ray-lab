from flask import render_template, request
from flask import redirect, url_for
import os
from PIL import Image
from app.utils import pipeline_model

UPLOAD_FLODER = 'static/uploads'

def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def xraylab():
    return render_template('xraylab.html')


def predictions():

    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        # prediction (pass to pipeline model)
        pipeline_model(path,filename)


        return render_template('predictions.html',fileupload=True,img_name=filename)


    return render_template('predictions.html',fileupload=False)
