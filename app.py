from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil input dari form
            input_values = [float(request.form[f'input{i}']) for i in range(1, 7)]
            input_array = np.array([input_values]).reshape(1, -1)

            # Prediksi
            prediction = model.predict(input_array)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
