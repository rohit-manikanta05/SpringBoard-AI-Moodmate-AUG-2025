# MoodMate: Emotion Detection + Music Recommendation (Starter Project)

This starter kit gives you a clean, step-by-step implementation that matches your project plan.
You will:
1) Train a CNN on FER-2013 for facial emotion recognition.
2) Build a content-based music recommender using TF‑IDF over tags/genres/moods.
3) Run a Streamlit app that detects emotion from **image** or **text** and suggests tracks.

---

## 0) Setup

```bash
# (Recommended) Create a fresh virtual env
python -m venv .venv
# or: conda create -n moodmate python=3.10 -y && conda activate moodmate

# Activate it
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

Place **FER-2013** CSV at:
```
data/fer2013/fer2013.csv
```
(You can download FER-2013 from Kaggle.)

Place your **music dataset** CSV at:
```
data/music/songs.csv
```
Columns expected: `track_id,title,artist,genre,tags,mood`.  
A small sample is included—you should replace it with your real catalog for best results.

---

## 1) Train the Emotion CNN

```bash
python src/train_cnn_fer2013.py --epochs 20 --batch_size 128
```
This saves the model to `models/fer_cnn.keras` and label mapping to `models/class_names.json`.

---

## 2) Build the Recommender Index

```bash
python src/recommender/build_recommender.py
```
This creates `models/tfidf_vectorizer.joblib`, `models/song_index.joblib`, and a cleaned copy of your songs at `models/songs_clean.parquet`.

---

## 3) Run the App

```bash
streamlit run app.py
```
- **Image tab**: upload a face photo → predicts emotion → shows top 10 tracks.
- **Text tab**: type how you feel → predicts emotion (VADER + small lexicon) → shows tracks.

---

## Tips
- Replace `data/music/songs.csv` with your real playlist export (Spotify/Last.fm/CSV).
- You can map each emotion to slightly different keywords in `src/recommender/emotion_mapping.py`.
- For better accuracy on faces, ensure images are clear, front-facing, and well-lit.
