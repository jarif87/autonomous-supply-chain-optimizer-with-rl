import streamlit as st
import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pathlib

# Custom CSS for rounder form and eye-catching design
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #1e3d66;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        font-size: 1.5em;
        margin-top: 20px;
    }
    .form-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stTextInput > div > input, .stNumberInput > div > input, .stSelectbox > div > select {
        border-radius: 25px;
        border: 1px solid #dcdcdc;
        padding: 12px 15px;
        background-color: #f9f9f9;
        transition: all 0.3s;
    }
    .stTextInput > div > input:focus, .stNumberInput > div > input:focus, .stSelectbox > div > select:focus {
        border-color: #1e3d66;
        box-shadow: 0 0 5px rgba(30, 61, 102, 0.3);
        outline: none;
    }
    .stButton > button {
        background-color: #1e3d66;
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        font-size: 1.1em;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
        display: block;
        margin: 20px auto;
    }
    .stButton > button:hover {
        background-color: #2c3e50;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:active {
        background-color: #1a3c5e;
    }
    .explanation {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1e3d66;
        margin-bottom: 20px;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .log-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dcdcdc;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("Supply Chain Shipping Mode Predictor")

# Explanation section
st.header("About This App")
st.markdown("""
<div class="explanation">
This application predicts the optimal shipping mode for supply chain orders using a machine learning model (PPO). 
It takes order details like shipping days, costs, and locations to recommend the best shipping method 
(First Class, Same Day, Second Class, or Standard Class) to minimize delays, reduce late delivery risk, and maximize profit.

**How to Use the App:**
1. **Enter Order Details**: Fill in the form below with values in their natural units (e.g., costs in dollars, latitude/longitude in degrees, shipping delay in days).
2. **Ensure Valid Inputs**: All fields are required. The app automatically scales costs and latitude as needed.
3. **Click Predict**: Use the "Predict" button to get the recommended shipping mode and metrics.
4. **View Results**: See the predicted shipping mode, adjusted delay, risk, benefit, and reward score.
5. **Check Prediction Log**: View the log of all predictions at the bottom.
</div>
""", unsafe_allow_html=True)

# Define label encoding mappings
shipping_mode_mapping = {0: 'First Class', 1: 'Same Day', 2: 'Second Class', 3: 'Standard Class'}
market_mode_mapping = {0: 'Africa', 1: 'Europe', 2: 'LATAM', 3: 'Pacific Asia', 4: 'USCA'}
order_region_mapping = {0: 'Canada', 1: 'Caribbean', 2: 'Central Africa', 3: 'Central America', 
                       4: 'Central Asia', 5: 'East Africa', 6: 'East of USA', 7: 'Eastern Asia', 
                       8: 'Eastern Europe', 9: 'North Africa', 10: 'Northern Europe', 11: 'Oceania', 
                       12: 'South America', 13: 'South Asia', 14: 'South of USA', 15: 'Southeast Asia', 
                       16: 'Southern Africa', 17: 'Southern Europe', 18: 'US Center', 19: 'West Africa', 
                       20: 'West Asia', 21: 'West of USA', 22: 'Western Europe'}
customer_city_mapping = {0: 'Aguadilla', 1: 'Alameda', 2: 'Albany', 3: 'Albuquerque', 4: 'Algonquin',
                        5: 'Alhambra', 6: 'Allentown', 7: 'Alpharetta', 8: 'Amarillo', 9: 'Anaheim',
                        10: 'Ann Arbor', 11: 'Annandale', 12: 'Annapolis', 13: 'Antioch', 14: 'Apex',
                        15: 'Apopka', 16: 'Arecibo', 17: 'Arlington', 18: 'Arlington Heights', 19: 'Asheboro',
                        20: 'Astoria', 21: 'Atlanta', 22: 'Augusta', 23: 'Aurora', 24: 'Austin',
                        25: 'Azusa', 26: 'Bakersfield', 27: 'Baldwin Park', 28: 'Ballwin', 29: 'Baltimore',
                        30: 'Bartlett', 31: 'Bay Shore', 32: 'Bayamon', 33: 'Bayonne', 34: 'Baytown',
                        35: 'Beaverton', 36: 'Bell Gardens', 37: 'Bellflower', 38: 'Bellingham', 39: 'Beloit',
                        40: 'Bend', 41: 'Bensalem', 42: 'Berwyn', 43: 'Billings', 44: 'Birmingham',
                        45: 'Bismarck', 46: 'Blacksburg', 47: 'Bloomfield'}
order_city_mapping = {0: 'Aachen', 1: 'Aalen', 2: 'Aalst', 3: 'Aba', 4: 'Abadan',
                     5: 'Abakaliki', 6: 'Abbeville', 7: 'Abbotsford', 8: 'Abeokuta', 9: 'Aberdeen',
                     10: 'Abha', 11: 'Abidjan', 12: 'Abilene', 13: 'Abreu e Lima', 14: 'Abu Kabir',
                     15: 'Acarigua', 16: 'Acayucan', 17: 'Accra', 18: 'Acerra', 19: 'Acireale',
                     20: 'Acuna', 21: 'Acambaro', 22: 'Ad Diwaniyah', 23: 'Ad Diwem', 24: 'Adana'}

# Define string options for user input
shipping_mode_options = list(shipping_mode_mapping.values())
market_options = list(market_mode_mapping.values())
order_region_options = list(order_region_mapping.values())
customer_city_options = list(customer_city_mapping.values())
order_city_options = list(order_city_mapping.values())

# Load encoded DataFrame and mappings
@st.cache_data
def load_data_and_mappings():
    file_path = pathlib.Path("encoded_df_and_mappings.pkl")
    if not file_path.exists():
        st.error(f"❌ 'encoded_df_and_mappings.pkl' not found at {file_path.absolute()}.")
        st.stop()
    try:
        with file_path.open('rb') as f:
            data = pickle.load(f)
        return data['data'], data['mappings']
    except Exception as e:
        st.error(f"❌ Error loading 'encoded_df_and_mappings.pkl': {str(e)}")
        st.stop()

df, mappings = load_data_and_mappings()
st.success("✅ Loaded encoded DataFrame and mappings")
mappings.update({'Shipping Mode': shipping_mode_mapping, 'Market': market_mode_mapping, 
                'Order Region': order_region_mapping, 'Customer City': customer_city_mapping, 
                'Order City': order_city_mapping})

# Load MinMaxScaler
@st.cache_data
def load_scaler():
    file_path = pathlib.Path("minmax_scaler.pkl")
    if not file_path.exists():
        st.error(f"❌ 'minmax_scaler.pkl' not found at {file_path.absolute()}.")
        st.stop()
    try:
        with file_path.open('rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"❌ Error loading 'minmax_scaler.pkl': {str(e)}")
        st.stop()

minmax_scaler = load_scaler()
st.success("✅ Loaded MinMaxScaler")

# Load trained PPO model
@st.cache_resource
def load_model():
    file_path = pathlib.Path("supply_chain_ppo_model.zip")
    if not file_path.exists():
        st.error(f"❌ 'supply_chain_ppo_model.zip' not found at {file_path.absolute()}.")
        st.stop()
    try:
        return PPO.load(file_path)
    except Exception as e:
        st.error(f"❌ Error loading 'supply_chain_ppo_model.zip': {str(e)}")
        st.stop()

model = load_model()
st.success("✅ Loaded PPO model")

# Define state and reward columns
state_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order',
              'Order Item Total', 'Product Price', 'Latitude', 'Longitude', 'Shipping_Delay',
              'Late_delivery_risk', 'Shipping Mode_Encoded', 'Market_Encoded',
              'Order Region_Encoded', 'Customer City_Encoded', 'Order City_Encoded']
reward_cols = ['Shipping_Delay', 'Late_delivery_risk', 'Benefit per order']
numerical_cols = ['Benefit per order', 'Order Item Total', 'Product Price', 'Latitude']
categorical_cols = ['Shipping Mode', 'Market', 'Order Region', 'Customer City', 'Order City']

# Preprocess state
def preprocess_state(df_row, state_cols, numerical_cols, minmax_scaler, mappings):
    df_row = df_row.copy()
    for col in categorical_cols:
        encoded_col = f"{col}_Encoded"
        if encoded_col in state_cols:
            categories = mappings.get(col, {})
            try:
                df_row[encoded_col] = df_row[col].apply(
                    lambda x: list(categories.keys())[list(categories.values()).index(x)] 
                    if x in categories.values() else 0)
            except ValueError as e:
                st.warning(f"⚠️ Unrecognized value in {col}: {df_row[col].iloc[0]}. Defaulting to 0.")
                df_row[encoded_col] = 0
    
    missing_num_cols = [col for col in numerical_cols if col not in df_row.columns]
    if missing_num_cols:
        st.error(f"❌ Missing numerical columns: {', '.join(missing_num_cols)}")
        st.stop()
    
    try:
        numerical_data = df_row[numerical_cols].astype(float)
        scaled_data = minmax_scaler.transform(numerical_data)
        df_row[numerical_cols] = scaled_data
    except ValueError as e:
        st.error(f"❌ Invalid numerical values in {numerical_cols}: {str(e)}")
        st.stop()
    
    missing_state_cols = [col for col in state_cols if col not in df_row.columns]
    if missing_state_cols:
        st.error(f"❌ Missing state columns: {', '.join(missing_state_cols)}")
        st.stop()
    
    state = df_row[state_cols].astype(float).values
    return np.clip(state, -np.inf, np.inf)  # Avoid clipping to 0-1 for non-scaled inputs

# Map action to shipping mode
def action_to_shipping_mode(action, mappings):
    return mappings['Shipping Mode'].get(int(action), 'Unknown')

# Predict shipping mode
def predict_shipping_mode(df_new, mappings, state_cols, numerical_cols, minmax_scaler, model):
    required_cols = list(set([col for col in state_cols if not col.endswith('_Encoded')] + reward_cols))
    missing_cols = [col for col in required_cols if col not in df_new.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()
    
    for col in required_cols:
        if df_new[col].isna().any():
            st.error(f"❌ NaN values detected in column '{col}'")
            st.stop()
    
    predictions = []
    try:
        states = preprocess_state(df_new, state_cols, numerical_cols, minmax_scaler, mappings)
        actions, _ = model.predict(states, deterministic=True)
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.stop()
    
    progress_bar = st.progress(0)
    for idx, action in enumerate(actions):
        row = df_new.iloc[idx]
        try:
            shipping_delay = float(row['Shipping_Delay'])
            late_risk = float(row['Late_delivery_risk'])
            benefit = float(row['Benefit per order'])
        except (ValueError, TypeError) as e:
            st.error(f"❌ Invalid numerical value in row {idx}: {str(e)}")
            st.stop()
        
        shipping_mode = action_to_shipping_mode(action, mappings)
        delay_factor = [1.0, 0.95, 0.90, 0.85][int(action)]
        risk_factor = [1.0, 0.95, 0.90, 0.85][int(action)]
        shipping_delay = np.clip(shipping_delay * delay_factor, 0.0, np.inf)
        late_risk = np.clip(late_risk * risk_factor, 0.0, 1.0)
        reward = -0.1 * shipping_delay - 0.2 * late_risk + 2.0 * benefit
        action_penalty = [0.0, -0.02, -0.04, -0.06][int(action)]
        reward += action_penalty
        if shipping_delay < 0.4:
            reward += 0.75
        if late_risk < 0.4:
            reward += 0.75
        
        predictions.append({
            'row_index': idx, 'action': int(action), 'shipping_mode': shipping_mode,
            'shipping_delay': float(shipping_delay), 'late_risk': float(late_risk),
            'benefit': float(benefit), 'reward': float(reward)
        })
        
        log_file = pathlib.Path('predictions.log')
        try:
            with log_file.open('a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp},{idx},{action},{shipping_mode},{shipping_delay:.3f},{late_risk:.3f},{benefit:.3f},{reward:.3f}\n")
        except Exception as e:
            st.warning(f"⚠️ Failed to write to log file: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(df_new))
    
    return pd.DataFrame(predictions)

# Input section
st.header("Input New Data")
st.markdown("<div class='form-container'>", unsafe_allow_html=True)
input_data = {}
required_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order',
                'Order Item Total', 'Product Price', 'Latitude', 'Longitude', 'Shipping_Delay',
                'Late_delivery_risk', 'Shipping Mode', 'Market', 'Order Region', 'Customer City', 'Order City']
default_values = {'Days for shipping (real)': 0.0, 'Days for shipment (scheduled)': 0.0, 
                 'Benefit per order': 100.0,  # Dollars, unscaled
                 'Order Item Total': 200.0,   # Dollars, unscaled
                 'Product Price': 50.0,       # Dollars, unscaled
                 'Latitude': 40.0,            # Degrees, unscaled
                 'Longitude': -74.0,          # Degrees, unscaled
                 'Shipping_Delay': 2.0,       # Days, unscaled
                 'Late_delivery_risk': 0.5,   # Probability, 0-1
                 'Shipping Mode': 'Standard Class', 'Market': 'USCA', 
                 'Order Region': 'US Center', 'Customer City': 'Atlanta', 'Order City': 'Aachen'}

for col in required_cols:
    if col in ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order',
               'Order Item Total', 'Product Price', 'Latitude', 'Longitude', 'Shipping_Delay', 'Late_delivery_risk']:
        min_value = -90.0 if col == 'Latitude' else -180.0 if col == 'Longitude' else 0.0
        max_value = 90.0 if col == 'Latitude' else 180.0 if col == 'Longitude' else 10000.0 if col in ['Benefit per order', 'Order Item Total', 'Product Price', 'Shipping_Delay'] else 1.0 if col == 'Late_delivery_risk' else 100.0
        input_data[col] = st.number_input(
            f"{col}",
            min_value=min_value,
            max_value=max_value,
            value=default_values[col], 
            key=f"input_{col.replace(' ', '_')}")
    elif col == 'Shipping Mode':
        input_data[col] = st.selectbox(f"{col}", options=shipping_mode_options, 
                                       index=shipping_mode_options.index(default_values[col]), 
                                       key=f"input_{col.replace(' ', '_')}")
    elif col == 'Market':
        input_data[col] = st.selectbox(f"{col}", options=market_options, 
                                       index=market_options.index(default_values[col]), 
                                       key=f"input_{col.replace(' ', '_')}")
    elif col == 'Order Region':
        input_data[col] = st.selectbox(f"{col}", options=order_region_options, 
                                       index=order_region_options.index(default_values[col]), 
                                       key=f"input_{col.replace(' ', '_')}")
    elif col == 'Customer City':
        input_data[col] = st.selectbox(f"{col}", options=customer_city_options, 
                                       index=customer_city_options.index(default_values[col]), 
                                       key=f"input_{col.replace(' ', '_')}")
    elif col == 'Order City':
        input_data[col] = st.selectbox(f"{col}", options=order_city_options, 
                                       index=order_city_options.index(default_values[col]), 
                                       key=f"input_{col.replace(' ', '_')}")

missing_cols = [col for col in required_cols if input_data.get(col) is None]
valid_input = len(missing_cols) == 0

# Center the Predict button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict", disabled=not valid_input):
    if not valid_input:
        st.warning(f"⚠️ Please fill in: {', '.join(missing_cols)}")
    else:
        try:
            df_new = pd.DataFrame([input_data])
            with st.spinner("Predicting..."):
                predictions_df = predict_shipping_mode(df_new, mappings, state_cols, numerical_cols, minmax_scaler, model)
            st.subheader("Prediction Results")
            st.write(predictions_df[['action', 'shipping_mode', 'shipping_delay', 'late_risk', 'benefit', 'reward']])
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # Close form-container div

# Display log file
log_file = pathlib.Path('predictions.log')
if log_file.exists():
    st.header("Prediction Log")
    st.markdown("<div class='log-section'>", unsafe_allow_html=True)
    try:
        with log_file.open('r') as f:
            st.text(f.read())
    except Exception as e:
        st.warning(f"⚠️ Failed to read log file: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)