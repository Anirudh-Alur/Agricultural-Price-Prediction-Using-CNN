from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import save_model

app = Flask(__name__)
model = load_model('models/veg_price_predictor.h5')
X_scaler = joblib.load('models/x_scaler.pkl')
y_scaler = joblib.load('models/y_scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'saved_model'
PLOT_PATH = 'static/training_plot.png'

df = pd.read_csv('Veg_Market.csv')

def predict_next_7_days(commodity, state, district, market, start_date):
    print("Inside predict_next_7_days")
    print("Inputs:", commodity, state, district, market, start_date)

    predictions = []

    for i in range(1, 8):
        try:
            date = start_date + timedelta(days=i)
            print(f"Predicting for date: {date.strftime('%Y-%m-%d')}")

            features = {
                'Commodity_Code': encoders['Commodity'].transform([commodity])[0],
                'State_Code': encoders['State'].transform([state])[0],
                'District_Code': encoders['District'].transform([district])[0],
                'Market_Code': encoders['Market'].transform([market])[0],
                'Year': date.year,
                'Month': date.month,
                'Day': date.day,
                'DayOfWeek': date.weekday(),
                'DayOfYear': date.timetuple().tm_yday
            }

            feature_array = np.array([list(features.values())])
            feature_scaled = X_scaler.transform(feature_array)
            feature_reshaped = feature_scaled.reshape(1, 1, -1)

            pred_scaled = model.predict(feature_reshaped, verbose=0)
            pred = y_scaler.inverse_transform(pred_scaled)[0]

            predictions.append({
                'Date': date.strftime('%d-%m-%Y'),
                'Minimum': round(pred[0], 2),
                'Maximum': round(pred[1], 2),
                'Average': round(pred[2], 2)
            })

        except Exception as e:
            print(f"Error predicting for {date.strftime('%d-%m-%Y')}: {e}")
            continue  # Skip that day's prediction

    print("Finished predictions:", predictions)
    return predictions





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def home():
    return render_template('index.html')



@app.route('/upload_train', methods=['GET', 'POST'])
def upload_train():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # --- Step 1: Load and preprocess data ---
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfYear'] = df['Date'].dt.dayofyear

            encoders = {}
            for col in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
                encoders[col] = LabelEncoder()
                df[f'{col}_Code'] = encoders[col].fit_transform(df[col].astype(str))

            features = ['Commodity_Code', 'State_Code', 'District_Code', 'Market_Code', 
                        'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
            X = df[features].values
            y = df[['Minimum', 'Maximum', 'Average']].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)

            X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

            # --- Step 2: Build model ---
            model = Sequential([
                Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, len(features))),
                MaxPooling1D(pool_size=1),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(3)
            ])

            model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

            # --- Step 3: Train ---
            history = model.fit(
                X_train_reshaped, y_train_scaled,
                epochs=25,
                batch_size=4,
                validation_data=(X_test_reshaped, y_test_scaled),
                verbose=1
            )

            # --- Step 4: Save model ---
            model.save(os.path.join(MODEL_FOLDER, 'model.h5'))

            # --- Step 5: Plot and save graph ---
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOT_PATH)
            plt.close()

            return render_template('trainn_upload.html', image_path=PLOT_PATH)

    return render_template('trainn_upload.html')


@app.route('/predict')
def predict():
    states = df['State'].unique()
    districts = df[['State', 'District']].drop_duplicates().to_dict('records')
    markets = df[['State', 'District', 'Market']].drop_duplicates().to_dict('records')
    commodities = df[['State', 'District', 'Market', 'Commodity']].drop_duplicates().to_dict('records')

    return render_template(
        'predict.html',
        states=states,
        districts=districts,
        markets=markets,
        commodities=commodities
    )


@app.route('/upload_file')
def upload_file():
    return render_template('trainn_upload.html')


@app.route('/project_working')
def project_working():
    return render_template('project_working.html')

@app.route('/predict_result', methods=['GET', 'POST'])
def predict_result():
    if request.method == 'POST':
        try:
            commodity = request.form['commodity']
            state = request.form['state']
            district = request.form['district']
            market = request.form['market']
            start_date_str = request.form['start_date']
            
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            
            print("Form submitted with:", commodity, state, district, market, start_date)
            
            predictions = predict_next_7_days(commodity, state, district, market, start_date)
            
            # Convert numpy float32 values to regular Python floats for JSON serialization
            for pred in predictions:
                pred['Minimum'] = float(pred['Minimum'])
                pred['Maximum'] = float(pred['Maximum'])
                pred['Average'] = float(pred['Average'])
            
            print("Predictions processed:", predictions)
            states = df['State'].unique()
            districts = df[['State', 'District']].drop_duplicates().to_dict('records')
            markets = df[['State', 'District', 'Market']].drop_duplicates().to_dict('records')
            commodities = df[['State', 'District', 'Market', 'Commodity']].drop_duplicates().to_dict('records')   
            return render_template(
                'predict.html',
                predictions=predictions,
                commodity=commodity,
                state=state,
                district=district,
                market=market,
                start_date=start_date_str,
                states=states,
                districts=districts,
                markets=markets,
                commodities=commodities
            )
        except Exception as e:
            print("Error occurred:", e)
            import traceback
            traceback.print_exc()  # Print detailed error information
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')




if __name__ == '__main__':
    app.run(debug=True)