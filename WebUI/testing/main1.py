from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Sample JSON data
    data = {
        "name": "John",
        "age": 30,
        "city": "New York"
    }
    return render_template('index1.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)