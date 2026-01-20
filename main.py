from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/doctor/<path:filename>')
def serve_doctor(filename):
    doctor_folder = os.path.join(app.root_path, 'doctor')
    return send_from_directory(doctor_folder, filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)