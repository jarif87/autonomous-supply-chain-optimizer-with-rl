# Autonomous Supply Chain Optimizer with RL

The Supply Chain Shipping Mode Predictor is a Streamlit-based web application that uses a Proximal Policy Optimization (PPO) reinforcement learning model to recommend optimal shipping modes (First Class, Same Day, Second Class, Standard Class) for supply chain orders. Built with a custom Gym environment (SupplyChainEnv), it optimizes for minimal shipping delays, reduced late delivery risk and maximum profit, achieving a 100% success rate (reward > 0). The app features a modern UI with rounded forms and buttons, styled with custom CSS and supports unscaled user inputs (e.g., costs in dollars, latitude in degrees) with automated preprocessing using MinMaxScaler.

### Features
- Reinforcement Learning: Utilizes a PPO model trained in a custom SupplyChainEnv to predict shipping modes based on order data.
- User-Friendly Interface: Clean, responsive UI with rounded input fields and buttons, built with Streamlit and custom CSS.
- Data Preprocessing: Handles unscaled inputs (e.g., Benefit per order, Product Price) using a pre-trained MinMaxScaler.
- Error Handling: Robust validation for file loading, categorical encoding, and prediction logging.
- Prediction Logging: Stores prediction history with timestamps in a predictions.log file.

### TechnologiesPython
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Stable-Baselines3
- Gym
- HTML/CSS

### Installation
- Clone the Repository:
```
git clone https://github.com/jarif87/autonomous-supply-chain-optimizer-with-rl.git
cd streamlit app
```
- Install Dependencies:
- Ensure Python 3.8+ is installed, then run:
```
pip install -r requirements.txt
```
- Required packages: streamlit, pandas, numpy, scikit-learn, stable-baselines3, gym.
### Download Model Files:
- Place the following pre-trained files in the project root:
    - encoded_df_and_mappings.pkl: Encoded dataset and mappings.
    - minmax_scaler.pkl: Pre-trained MinMaxScaler.
    - supply_chain_ppo_model.zip: Trained PPO model.

### Usage
- Run the Application:bash
```
streamlit run app.py

```
### Enter Order Details:
- Input numerical values in their natural units (e.g., costs in dollars, latitude/longitude in degrees, shipping delay in days).
- Select categorical options (e.g., Shipping Mode, Market) from dropdowns.

### Predict:
- Click the centered "Predict" button to view the recommended shipping mode, adjusted delay, risk, benefit, and reward score.
- View Prediction Log:
    - Scroll to the "Prediction Log" section to see a history of predictions stored in predictions.log.

### Model Performance:
- Average Episode Reward: 175.119 ± 14.160
- Success Rate (Reward > 0): 100%
- Action Distribution: Predominantly Second Class (1912/2000 actions)

- Preprocessing: Scales four features (Benefit per order, Order Item Total, Product Price, Latitude) using MinMaxScaler, while other inputs (e.g., Shipping Delay, Longitude) are used as-is.
- UI Design: Features rounded input fields and buttons (border-radius: 25px), a blue-themed color scheme (#1e3d66), and responsive styling.

