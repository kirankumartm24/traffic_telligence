import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("trafficvolume.csv")
df['temp'] = df['temp'].fillna(df['temp'].mean())
df['rain'] = df['rain'].fillna(df['rain'].mean())
df['snow'] = df['snow'].fillna(df['snow'].mean())
df['weather'] = df['weather'].fillna('Clouds')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
df['day'] = df['date'].dt.dayofweek  
df['month'] = df['date'].dt.month
encoders = {}
le = LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
encoders['weather'] = le
X = df[['hour', 'weather', 'temp', 'day', 'month']]
y = df['traffic_volume']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "traffic_model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")