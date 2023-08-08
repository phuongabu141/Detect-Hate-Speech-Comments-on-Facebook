from application import app

# python run.py
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
