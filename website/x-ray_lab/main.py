from flask import Flask
from app import views

app = Flask(__name__)

# url
#app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/xraylab','xraylab',views.xraylab)
app.add_url_rule('/xraylab/predictions','predictions',views.predictions,methods=['GET','POST'])


if __name__ == "__main__":
    app.run(debug=True)
