
from flask import Flask, render_template
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    stock = "TCS.NS"
    data = yf.download(stock, period="6mo")
    close_data = data[['Close']]

    if len(close_data) < 60:
        return "Not enough data to predict."

    scaled_data = scaler.transform(close_data)
    last_60 = scaled_data[-60:]
    last_60 = np.reshape(last_60, (1,60,1))

    prediction = model.predict(last_60)
    predicted_price = scaler.inverse_transform(prediction)
    last_price = close_data.iloc[-1][0]

    trend = "UP 📈" if predicted_price[0][0] > last_price else "DOWN 📉"

    return render_template("index.html",
                           price=round(predicted_price[0][0],2),
                           trend=trend,
                           last=round(last_price,2))

if __name__ == "__main__":
    app.run(debug=True)
