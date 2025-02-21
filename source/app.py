from flask import Flask,render_template,url_for
from flask_mysqldb import MySQL

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/live-mon')
def live_mon():
    return render_template("livemon.html")

@app.route('/new-crim')
def new_crim():
    return render_template("newcrim.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
