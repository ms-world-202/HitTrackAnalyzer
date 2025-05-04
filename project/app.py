from flask import Flask, render_template, request
import numpy as np
import pickle
import logging
import webbrowser
from threading import Timer

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load model and encoders
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("track_encoder.pkl", "rb") as f:
        track_encoder = pickle.load(f)
    with open("artist_mean_popularity.pkl", "rb") as f:
        artist_mean_popularity = pickle.load(f)
    with open("album_mean_popularity.pkl", "rb") as f:
        album_mean_popularity = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load model files: {str(e)}")
    raise

def validate_input(value, min_val, max_val, name):
    """Validate input ranges"""
    try:
        num = float(value)
        if not (min_val <= num <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return num
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {str(e)}")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate inputs
        duration_ms = int(request.form['duration_ms'])
        explicit = int(validate_input(request.form['explicit'], 0, 1, 'explicit'))
        danceability = validate_input(request.form['danceability'], 0, 1, 'danceability')
        energy = validate_input(request.form['energy'], 0, 1, 'energy')
        loudness = validate_input(request.form['loudness'], -60, 0, 'loudness')
        speechiness = validate_input(request.form['speechiness'], 0, 1, 'speechiness')
        acousticness = validate_input(request.form['acousticness'], 0, 1, 'acousticness')
        instrumentalness = validate_input(request.form['instrumentalness'], 0, 1, 'instrumentalness')
        liveness = validate_input(request.form['liveness'], 0, 1, 'liveness')
        valence = validate_input(request.form['valence'], 0, 1, 'valence')
        tempo = validate_input(request.form['tempo'], 50, 200, 'tempo')
        track_genre = request.form['track_genre']
        album_name = request.form['album_name']
        artists = request.form['artists']

        # Transform track genre
        if track_genre in track_encoder.classes_:
            track_genre_encoded = track_encoder.transform([track_genre])[0]
        else:
            track_genre_encoded = len(track_encoder.classes_)  # Default for unknown genre

        # Handle unknown artists and albums
        artist_pop = artist_mean_popularity.get(artists, np.mean(list(artist_mean_popularity.values())))
        album_pop = album_mean_popularity.get(album_name, np.mean(list(album_mean_popularity.values())))

        # Create feature array in correct order
        features = np.array([[
            duration_ms, explicit, danceability, energy, loudness,
            speechiness, acousticness, instrumentalness, liveness,
            valence, tempo, album_pop, artist_pop, track_genre_encoded
        ]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_score = round(prediction[0], 2)

        result = (f"The song is {'a Hit' if predicted_score >= 70 else 'Not a Hit'} "
                 f"(Score: {predicted_score})")

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        result = f"Error in prediction: {str(e)}"

    return render_template("result.html", result=result)

def open_browser():
    """Open browser automatically when app starts"""
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Print clear URL information
    print("\n" + "="*50)
    print("FLASK SERVER RUNNING")
    print("="*50)
    print("\nACCESS YOUR APP AT:")
    print("👉 http://127.0.0.1:5000 👈")
    print("\nPress CTRL+C to stop the server")
    print("="*50 + "\n")
    
    # Open browser automatically after 1 second delay
    Timer(1, open_browser).start()
    
    # Run the app with explicit settings
    app.run(host='0.0.0.0', port=5000, debug=True)