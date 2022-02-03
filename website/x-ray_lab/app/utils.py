import os
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# load the model
model = load_model('./model/xrayl_model1.h5',compile=False)

# settings
train_df = pd.read_csv("./dataset/train-small.csv")
img_dir = './dataset/images'
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
          'Mass', 'Nodule', 'Atelectasis','Pneumothorax','Pleural_Thickening',
          'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']


def get_mean_std_per_batch(df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        path = img_dir+'/' + img
        sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])

    return mean, std

def load_image_normalize(path, mean, std, H=320, W=320):
    x = image.load_img(path, target_size=(H, W))
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x

def plot_predictions(preds_float, img_name):
    pred_img = Image.open('./static/images/predictions_plain.png')
    draw = ImageDraw.Draw(pred_img)
    font = ImageFont.truetype("./static/fonts/Roboto-Regular.ttf", size=45)

    red = 'rgb(255, 0, 0)'
    green = 'rgb(0, 153, 0)'
    yellow = 'rgb(250, 176, 3)'

    x = 600
    y = 50
    for p in range(14):
        p_value = preds_float.item(0,p)
        if p_value >= 0.6:
            draw.text((x, y), "{:.0%}".format(p_value), fill=red, font=font)

        elif p_value < 0.5:
            draw.text((x, y), "{:.0%}".format(p_value), fill=green, font=font)

        else:
            draw.text((x, y), "{:.0%}".format(p_value), fill=yellow, font=font)
        y = y + 70

    pred_img.save('./static/predict/{}'.format(img_name))


def pipeline_model(path,filename):
    mean, std = get_mean_std_per_batch(train_df)
    im_path = path
    fn = filename
    processed_image = load_image_normalize(im_path, mean, std)
    preds = model.predict(processed_image)
    plot_predictions(preds, fn)
