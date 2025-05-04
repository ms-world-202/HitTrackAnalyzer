import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle

# Load and clean data
data = pd.read_csv(r"C:\Users\sidha\Downloads\spotify_dataset.csv")
data.drop(["Unnamed: 0", "track_id"], axis=1, inplace=True)

# Fill missing values
for col in ["loudness", "instrumentalness", "liveness", "valence"]:
    data[col].fillna(data[col].median(), inplace=True)

for col in ['artists', 'album_name', 'track_name']:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.drop_duplicates(inplace=True)

# Remove outliers
Q1, Q3 = data['duration_ms'].quantile([0.25, 0.75])
IQR = Q3 - Q1
data = data[(data['duration_ms'] >= Q1 - 1.5 * IQR) & 
            (data['duration_ms'] <= Q3 + 1.5 * IQR)]

# Drop unnecessary columns
data.drop(['key', 'mode', 'time_signature', 'track_name'], axis=1, inplace=True)

# Target Encoding
artist_mean_popularity = data.groupby('artists')['popularity'].mean().to_dict()
album_mean_popularity = data.groupby('album_name')['popularity'].mean().to_dict()

data['artists'] = data['artists'].map(artist_mean_popularity)
data['album_name'] = data['album_name'].map(album_mean_popularity)

# Label Encoding
track_encoder = LabelEncoder()
data['track_genre'] = track_encoder.fit_transform(data['track_genre'].astype(str))

# Prepare features - ORDER MATTERS!
features = [
    'duration_ms', 'explicit', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'album_name', 'artists', 'track_genre'
]
X = data[features]
y = data["popularity"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")

# Save artifacts
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("track_encoder.pkl", "wb") as f:
    pickle.dump(track_encoder, f)

with open("artist_mean_popularity.pkl", "wb") as f:
    pickle.dump(artist_mean_popularity, f)

with open("album_mean_popularity.pkl", "wb") as f:
    pickle.dump(album_mean_popularity, f)

print("Model training complete. Artifacts saved.")