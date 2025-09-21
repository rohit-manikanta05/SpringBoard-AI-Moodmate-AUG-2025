# import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
# from PIL import Image
# import cv2
# import tensorflow as tf

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from src.utils.image import detect_and_crop_face
# from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

# st.set_page_config(page_title="MoodMate", page_icon="🎵")

# ### 🔧 PATCH: Add missing imports + define TextCleaner
# import re
# from sklearn.base import BaseEstimator, TransformerMixin

# class TextCleaner(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None): 
#         return self
#     def transform(self, X):
#         cleaned = []
#         for t in X:
#             t = str(t).lower()
#             t = re.sub(r"[^a-z0-9\s]+", " ", t)   # keep alphanum + space
#             t = re.sub(r"\s+", " ", t).strip()    # normalize whitespace
#             cleaned.append(t)
#         return cleaned
# ### 🔧 END PATCH


# # --- Load CNN model (if present) ---
# MODEL_PATH = os.path.join("models", "fer_cnn.keras")
# CLASS_JSON = os.path.join("models", "class_names.json")

# cnn_model = None
# class_names = None
# if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
#     cnn_model = tf.keras.models.load_model(MODEL_PATH)
#     with open(CLASS_JSON) as f:
#         class_names = json.load(f)

# # --- Load recommender artifacts ---
# VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
# INDEX_PATH = os.path.join("models", "song_index.joblib")
# SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

# vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
# X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
# songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

# def recommend_for_emotion(emotion_name, top_k=10):
#     if vec is None or X is None or songs_df is None:
#         return pd.DataFrame(columns=["title","artist","genre","mood","tags","search_query"])
#     query = EMOTION_QUERY.get(emotion_name, "chill balanced")
#     qvec = vec.transform([query])
#     sims = (qvec @ X.T).toarray().ravel()
#     idx = np.argsort(-sims)[:top_k]
#     res = songs_df.iloc[idx][["title","artist","genre","mood","tags","search_query"]].copy()
#     res["score"] = sims[idx]
#     return res

# def predict_emotion_from_face(image_bgr):
#     if cnn_model is None or class_names is None:
#         return None, None
#     crop = detect_and_crop_face(image_bgr)
#     x = np.expand_dims(crop, axis=0)
#     probs = cnn_model.predict(x, verbose=0)[0]
#     pred_id = int(np.argmax(probs))
#     return class_names[pred_id], float(np.max(probs))

# def predict_emotion_from_text(text):
#     # Hybrid: VADER + tiny keyword cues for specific emotions
#     analyzer = SentimentIntensityAnalyzer()
#     s = analyzer.polarity_scores(text)
#     compound = s["compound"]
#     text_l = text.lower()

#     cues = {
#         "angry": ["furious","angry","rage","annoyed","irritated","mad"],
#         "fear": ["afraid","scared","terrified","nervous","worried","anxious"],
#         "disgust": ["disgust","gross","nasty","revolting","repulsed"],
#         "sad": ["sad","depressed","down","unhappy","miserable","blue","cry"],
#         "surprise": ["surprised","shocked","astonished","amazed","wow"],
#         "happy": ["happy","joyful","glad","excited","delighted","great"],
#     }

#     # Keyword override
#     for emo, kw in cues.items():
#         if any(k in text_l for k in kw):
#             return emo, 0.9

#     if compound >= 0.5:
#         return "happy", compound
#     elif compound <= -0.6:
#         # choose between sad/angry via intensity of "!" etc.
#         if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
#             return "angry", abs(compound)
#         return "sad", abs(compound)
#     else:
#         return "neutral", 1.0 - abs(compound)

# st.title("🎵 MoodMate — Emotion → Music")
# st.caption("Detect emotion from a face photo or text, then get a mood-aligned playlist.")

# tab_img, tab_txt = st.tabs(["📷 From Image", "✍️ From Text"])

# with tab_img:
#     st.subheader("Upload a face photo")
#     img_file = st.file_uploader("Image file", type=["jpg","jpeg","png"])
#     if img_file is not None:
#         image_bytes = img_file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("Model not found. Please train the CNN first.")
#         else:
#             st.success(f"Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             if recs.empty:
#                 st.info("Recommender index missing. Run the recommender builder script.")
#             else:
#                 st.dataframe(recs)

# with tab_txt:
#     st.subheader("Describe how you feel")
#     txt = st.text_area("Type a sentence or two...", "")
#     if st.button("Analyze & Recommend", type="primary"):
#         if not txt.strip():
#             st.warning("Please enter some text.")
#         else:
#             emo, conf = predict_emotion_from_text(txt)
#             st.success(f"Detected emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             if recs.empty:
#                 st.info("Recommender index missing. Run the recommender builder script.")
#             else:
#                 st.dataframe(recs)













# import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
# from PIL import Image
# import cv2
# import tensorflow as tf

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from src.utils.image import detect_and_crop_face
# from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

# st.set_page_config(page_title="MoodMate", page_icon="🎵")

# ### 🔧 PATCH: Add missing imports + define TextCleaner
# import re
# from sklearn.base import BaseEstimator, TransformerMixin

# class TextCleaner(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None): 
#         return self
#     def transform(self, X):
#         cleaned = []
#         for t in X:
#             t = str(t).lower()
#             t = re.sub(r"[^a-z0-9\s]+", " ", t)   # keep alphanum + space
#             t = re.sub(r"\s+", " ", t).strip()    # normalize whitespace
#             cleaned.append(t)
#         return cleaned
# ### 🔧 END PATCH


# # --- Load CNN model (if present) ---
# MODEL_PATH = os.path.join("models", "fer_cnn.keras")
# CLASS_JSON = os.path.join("models", "class_names.json")

# cnn_model = None
# class_names = None
# if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
#     cnn_model = tf.keras.models.load_model(MODEL_PATH)
#     with open(CLASS_JSON) as f:
#         class_names = json.load(f)

# # --- Load recommender artifacts ---
# VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
# INDEX_PATH = os.path.join("models", "song_index.joblib")
# SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

# vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
# X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
# songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

# def recommend_for_emotion(emotion_name, top_k=10):
#     if vec is None or X is None or songs_df is None:
#         return pd.DataFrame(columns=["title","artist","genre","mood","tags","search_query"])
#     query = EMOTION_QUERY.get(emotion_name, "chill balanced")
#     qvec = vec.transform([query])
#     sims = (qvec @ X.T).toarray().ravel()
#     idx = np.argsort(-sims)[:top_k]
#     res = songs_df.iloc[idx][["title","artist","genre","mood","tags","search_query"]].copy()
#     res["score"] = sims[idx]
#     return res

# def predict_emotion_from_face(image_bgr):
#     if cnn_model is None or class_names is None:
#         return None, None
#     crop = detect_and_crop_face(image_bgr)
#     x = np.expand_dims(crop, axis=0)
#     probs = cnn_model.predict(x, verbose=0)[0]
#     pred_id = int(np.argmax(probs))
#     return class_names[pred_id], float(np.max(probs))

# def predict_emotion_from_text(text):
#     # Hybrid: VADER + tiny keyword cues for specific emotions
#     analyzer = SentimentIntensityAnalyzer()
#     s = analyzer.polarity_scores(text)
#     compound = s["compound"]
#     text_l = text.lower()

#     cues = {
#         "angry": ["furious","angry","rage","annoyed","irritated","mad"],
#         "fear": ["afraid","scared","terrified","nervous","worried","anxious"],
#         "disgust": ["disgust","gross","nasty","revolting","repulsed"],
#         "sad": ["sad","depressed","down","unhappy","miserable","blue","cry"],
#         "surprise": ["surprised","shocked","astonished","amazed","wow"],
#         "happy": ["happy","joyful","glad","excited","delighted","great"],
#     }

#     # Keyword override
#     for emo, kw in cues.items():
#         if any(k in text_l for k in kw):
#             return emo, 0.9

#     if compound >= 0.5:
#         return "happy", compound
#     elif compound <= -0.6:
#         # choose between sad/angry via intensity of "!" etc.
#         if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
#             return "angry", abs(compound)
#         return "sad", abs(compound)
#     else:
#         return "neutral", 1.0 - abs(compound)

# # --- UI ---
# st.title("🎵 MoodMate — Emotion → Music")
# st.caption(
#     "Detect your emotion from a 📸 camera snapshot or 📂 uploaded photo, "
#     "or type a short text about how you feel to get a mood-aligned playlist 🎶"
# )

# tab_img, tab_txt = st.tabs(["📷 From Image", "✍️ From Text"])

# with tab_img:
#     st.subheader("📸 Take a snapshot or 📂 upload a face photo")

#     # --- Camera snapshot ---
#     cam_file = st.camera_input("Take a photo with your camera")
#     if cam_file is not None:
#         image = Image.open(cam_file).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("⚠️ Model not found. Please train the CNN first.")
#         else:
#             st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             if recs.empty:
#                 st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#             else:
#                 st.dataframe(recs)

#     # --- File uploader fallback ---
#     img_file = st.file_uploader("Or upload an image file", type=["jpg","jpeg","png"])
#     if img_file is not None:
#         image_bytes = img_file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("⚠️ Model not found. Please train the CNN first.")
#         else:
#             st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             if recs.empty:
#                 st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#             else:
#                 st.dataframe(recs)

# with tab_txt:
#     st.subheader("Describe how you feel")
#     txt = st.text_area("Type a sentence or two...", "")
#     if st.button("Analyze & Recommend", type="primary"):
#         if not txt.strip():
#             st.warning("⚠️ Please enter some text.")
#         else:
#             emo, conf = predict_emotion_from_text(txt)
#             st.success(f"😀 Detected emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             if recs.empty:
#                 st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#             else:
#                 st.dataframe(recs)












# import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
# from PIL import Image
# import cv2
# import tensorflow as tf

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from src.utils.image import detect_and_crop_face
# from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

# st.set_page_config(page_title="MoodMate", page_icon="🎵")

# ### 🔧 PATCH: Add missing imports + define TextCleaner
# import re
# from sklearn.base import BaseEstimator, TransformerMixin

# class TextCleaner(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None): 
#         return self
#     def transform(self, X):
#         cleaned = []
#         for t in X:
#             t = str(t).lower()
#             t = re.sub(r"[^a-z0-9\s]+", " ", t)   # keep alphanum + space
#             t = re.sub(r"\s+", " ", t).strip()    # normalize whitespace
#             cleaned.append(t)
#         return cleaned
# ### 🔧 END PATCH


# # --- Load CNN model (if present) ---
# MODEL_PATH = os.path.join("models", "fer_cnn.keras")
# CLASS_JSON = os.path.join("models", "class_names.json")

# cnn_model = None
# class_names = None
# if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
#     cnn_model = tf.keras.models.load_model(MODEL_PATH)
#     with open(CLASS_JSON) as f:
#         class_names = json.load(f)

# # --- Load recommender artifacts ---
# VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
# INDEX_PATH = os.path.join("models", "song_index.joblib")
# SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

# vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
# X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
# songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

# def recommend_for_emotion(emotion_name, top_k=10):
#     if vec is None or X is None or songs_df is None:
#         return pd.DataFrame(columns=["title","artist","genre","mood","tags","search_query"])
#     query = EMOTION_QUERY.get(emotion_name, "chill balanced")
#     qvec = vec.transform([query])
#     sims = (qvec @ X.T).toarray().ravel()
#     idx = np.argsort(-sims)[:top_k]
#     res = songs_df.iloc[idx][["title","artist","genre","mood","tags","search_query"]].copy()
#     res["score"] = sims[idx]
#     return res

# def predict_emotion_from_face(image_bgr):
#     if cnn_model is None or class_names is None:
#         return None, None
#     crop = detect_and_crop_face(image_bgr)
#     x = np.expand_dims(crop, axis=0)
#     probs = cnn_model.predict(x, verbose=0)[0]
#     pred_id = int(np.argmax(probs))
#     return class_names[pred_id], float(np.max(probs))

# def predict_emotion_from_text(text):
#     # Hybrid: VADER + tiny keyword cues for specific emotions
#     analyzer = SentimentIntensityAnalyzer()
#     s = analyzer.polarity_scores(text)
#     compound = s["compound"]
#     text_l = text.lower()

#     cues = {
#         "angry": ["furious","angry","rage","annoyed","irritated","mad"],
#         "fear": ["afraid","scared","terrified","nervous","worried","anxious"],
#         "disgust": ["disgust","gross","nasty","revolting","repulsed"],
#         "sad": ["sad","depressed","down","unhappy","miserable","blue","cry"],
#         "surprise": ["surprised","shocked","astonished","amazed","wow"],
#         "happy": ["happy","joyful","glad","excited","delighted","great"],
#     }

#     # Keyword override
#     for emo, kw in cues.items():
#         if any(k in text_l for k in kw):
#             return emo, 0.9

#     if compound >= 0.5:
#         return "happy", compound
#     elif compound <= -0.6:
#         if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
#             return "angry", abs(compound)
#         return "sad", abs(compound)
#     else:
#         return "neutral", 1.0 - abs(compound)


# # --- Song display helper (with fallback) ---
# def display_recommendations(recs):
#     if recs.empty:
#         st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#     else:
#         st.subheader("🎶 Recommended Songs")
#         for _, row in recs.iterrows():
#             title = row.get("title", "Unknown")
#             artist = row.get("artist", "Unknown")
#             query = row.get("search_query", f"{title} {artist}").replace(" ", "+")

#             st.markdown(f"**{title}** — {artist}")

#             try:
#                 youtube_embed = f"https://www.youtube.com/embed?listType=search&list={query}"
#                 st.markdown(
#                     f"""
#                     <iframe width="300" height="80"
#                     src="{youtube_embed}"
#                     frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
#                     </iframe>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             except Exception:
#                 youtube_url = f"https://www.youtube.com/results?search_query={query}"
#                 st.warning(f"⚠️ Could not embed player. [Search on YouTube]({youtube_url})")

#             st.markdown("---")


# # --- UI ---
# st.title("🎵 MoodMate — Emotion → Music")
# st.caption(
#     "Detect your emotion from a 📸 camera snapshot or 📂 uploaded photo, "
#     "or type a short text about how you feel to get a mood-aligned playlist 🎶"
# )

# tab_img, tab_txt = st.tabs(["📷 From Image", "✍️ From Text"])

# with tab_img:
#     st.subheader("📸 Take a snapshot or 📂 upload a face photo")

#     # --- Camera snapshot ---
#     cam_file = st.camera_input("Take a photo with your camera")
#     if cam_file is not None:
#         image = Image.open(cam_file).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("⚠️ Model not found. Please train the CNN first.")
#         else:
#             st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             display_recommendations(recs)

#     # --- File uploader fallback ---
#     img_file = st.file_uploader("Or upload an image file", type=["jpg","jpeg","png"])
#     if img_file is not None:
#         image_bytes = img_file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("⚠️ Model not found. Please train the CNN first.")
#         else:
#             st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             display_recommendations(recs)

# with tab_txt:
#     st.subheader("Describe how you feel")
#     txt = st.text_area("Type a sentence or two...", "")
#     if st.button("Analyze & Recommend", type="primary"):
#         if not txt.strip():
#             st.warning("⚠️ Please enter some text.")
#         else:
#             emo, conf = predict_emotion_from_text(txt)
#             st.success(f"😀 Detected emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=10)
#             display_recommendations(recs)










import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
import webbrowser
import urllib.parse
import streamlit.components.v1 as components

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.image import detect_and_crop_face
from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

st.set_page_config(page_title="MoodMate", page_icon="🎵")

### 🔧 PATCH: Add missing imports + define TextCleaner
import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        cleaned = []
        for t in X:
            t = str(t).lower()
            t = re.sub(r"[^a-z0-9\s]+", " ", t)   # keep alphanum + space
            t = re.sub(r"\s+", " ", t).strip()    # normalize whitespace
            cleaned.append(t)
        return cleaned
### 🔧 END PATCH

# --- Music Player Functions ---
def create_youtube_search_url(title, artist):
    """Create YouTube search URL for a song"""
    query = f"{title} {artist}".strip()
    encoded_query = urllib.parse.quote(query)
    return f"https://www.youtube.com/results?search_query={encoded_query}"

def create_spotify_search_url(title, artist):
    """Create Spotify search URL for a song"""
    query = f"{title} {artist}".strip()
    encoded_query = urllib.parse.quote(query)
    return f"https://open.spotify.com/search/{encoded_query}"

def create_youtube_embed_player(song_query, height=300):
    """Create an embedded YouTube player widget"""
    # Note: This creates a search-based embed. For actual video IDs, you'd need YouTube API
    search_query = urllib.parse.quote(song_query)
    
    html_code = f'''
    <div style="text-align: center; margin: 10px 0;">
        <iframe width="100%" height="{height}" 
                src="https://www.youtube.com/embed?listType=search&list={search_query}&autoplay=0" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
        </iframe>
    </div>
    '''
    return html_code

def display_playable_recommendations(recs_df, emotion_name):
    """Display recommendations with playback options"""
    if recs_df.empty:
        st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
        return
    
    st.success(f"🎵 Here are your {emotion_name}-based music recommendations!")
    
    # Add music service selection
    col1, col2 = st.columns([1, 1])
    with col1:
        music_service = st.selectbox(
            "Choose your preferred music service:",
            ["YouTube", "Spotify", "Embedded Player"],
            key=f"service_{emotion_name}"
        )
    
    with col2:
        auto_play_first = st.checkbox(
            "Auto-play first recommendation", 
            key=f"autoplay_{emotion_name}"
        )
    
    # Display recommendations with play buttons
    for idx, row in recs_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{row['title']}** by {row['artist']}")
                st.caption(f"Genre: {row['genre']} | Mood: {row['mood']}")
                if 'score' in row:
                    st.caption(f"Match Score: {row['score']:.3f}")
            
            with col2:
                if music_service == "YouTube":
                    youtube_url = create_youtube_search_url(row['title'], row['artist'])
                    if st.button(f"▶️ Play", key=f"yt_{idx}"):
                        webbrowser.open(youtube_url)
                        st.success(f"Opening {row['title']} on YouTube...")
                
                elif music_service == "Spotify":
                    spotify_url = create_spotify_search_url(row['title'], row['artist'])
                    if st.button(f"▶️ Play", key=f"sp_{idx}"):
                        webbrowser.open(spotify_url)
                        st.success(f"Opening {row['title']} on Spotify...")
            
            with col3:
                if st.button(f"📋 Copy", key=f"copy_{idx}"):
                    song_info = f"{row['title']} - {row['artist']}"
                    st.code(song_info)
                    st.success("Song info copied!")
            
            st.divider()
    
    # Embedded player option
    if music_service == "Embedded Player":
        st.subheader("🎵 Embedded Music Player")
        selected_song_idx = st.selectbox(
            "Select a song to play:",
            range(len(recs_df)),
            format_func=lambda x: f"{recs_df.iloc[x]['title']} - {recs_df.iloc[x]['artist']}"
        )
        
        selected_song = recs_df.iloc[selected_song_idx]
        song_query = f"{selected_song['title']} {selected_song['artist']}"
        
        if st.button("🎵 Load Player", key="load_player"):
            with st.spinner("Loading music player..."):
                player_html = create_youtube_embed_player(song_query)
                components.html(player_html, height=350)
    
    # Auto-play first song if enabled
    if auto_play_first and not recs_df.empty:
        first_song = recs_df.iloc[0]
        if music_service == "YouTube":
            url = create_youtube_search_url(first_song['title'], first_song['artist'])
        else:  # Spotify
            url = create_spotify_search_url(first_song['title'], first_song['artist'])
        
        st.info(f"🎵 Auto-playing: {first_song['title']} by {first_song['artist']}")
        st.markdown(f"[▶️ Click here to play]({url})")

# --- Load CNN model (if present) ---
MODEL_PATH = os.path.join("models", "fer_cnn.keras")
CLASS_JSON = os.path.join("models", "class_names.json")

cnn_model = None
class_names = None
if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON) as f:
        class_names = json.load(f)

# --- Load recommender artifacts ---
VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
INDEX_PATH = os.path.join("models", "song_index.joblib")
SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

def recommend_for_emotion(emotion_name, top_k=10):
    if vec is None or X is None or songs_df is None:
        return pd.DataFrame(columns=["title","artist","genre","mood","tags","search_query"])
    query = EMOTION_QUERY.get(emotion_name, "chill balanced")
    qvec = vec.transform([query])
    sims = (qvec @ X.T).toarray().ravel()
    idx = np.argsort(-sims)[:top_k]
    res = songs_df.iloc[idx][["title","artist","genre","mood","tags","search_query"]].copy()
    res["score"] = sims[idx]
    return res

def predict_emotion_from_face(image_bgr):
    if cnn_model is None or class_names is None:
        return None, None
    crop = detect_and_crop_face(image_bgr)
    x = np.expand_dims(crop, axis=0)
    probs = cnn_model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return class_names[pred_id], float(np.max(probs))

def predict_emotion_from_text(text):
    # Hybrid: VADER + tiny keyword cues for specific emotions
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text)
    compound = s["compound"]
    text_l = text.lower()

    cues = {
        "angry": ["furious","angry","rage","annoyed","irritated","mad"],
        "fear": ["afraid","scared","terrified","nervous","worried","anxious"],
        "disgust": ["disgust","gross","nasty","revolting","repulsed"],
        "sad": ["sad","depressed","down","unhappy","miserable","blue","cry"],
        "surprise": ["surprised","shocked","astonished","amazed","wow"],
        "happy": ["happy","joyful","glad","excited","delighted","great"],
    }

    # Keyword override
    for emo, kw in cues.items():
        if any(k in text_l for k in kw):
            return emo, 0.9

    if compound >= 0.5:
        return "happy", compound
    elif compound <= -0.6:
        # choose between sad/angry via intensity of "!" etc.
        if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
            return "angry", abs(compound)
        return "sad", abs(compound)
    else:
        return "neutral", 1.0 - abs(compound)

# --- UI ---
st.title("🎵 MoodMate — Emotion → Music Playback")
st.caption(
    "Detect your emotion from a 📸 camera snapshot or 📂 uploaded photo, "
    "or type a short text about how you feel to get mood-aligned songs that you can play instantly! 🎶"
)

# Add sidebar with music service info
with st.sidebar:
    st.header("🎵 Music Services")
    st.info(
        "**Available Options:**\n"
        "• **YouTube**: Opens songs in YouTube\n"
        "• **Spotify**: Opens songs in Spotify\n" 
        "• **Embedded Player**: Play directly in the app"
    )
    st.warning(
        "**Note**: For Spotify, you need to have the Spotify app installed or be logged into Spotify Web Player."
    )

tab_img, tab_txt, tab_cam = st.tabs(["📷 From Image", "✍️ From Text", "🎥 Real-time Webcam"])

with tab_img:
    st.subheader("📂 Upload a face photo")
    
    img_file = st.file_uploader("Or upload an image file", type=["jpg","jpeg","png"])
    if img_file is not None:
        image_bytes = img_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        emo, conf = predict_emotion_from_face(bgr)
        if emo is None:
            st.warning("⚠️ Model not found. Please train the CNN first.")
        else:
            st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
            recs = recommend_for_emotion(emo, top_k=5)
            display_playable_recommendations(recs, emo)

with tab_txt:
    st.subheader("Describe how you feel")
    txt = st.text_area("Type a sentence or two...", "")
    if st.button("Analyze & Get Playable Playlist", type="primary"):
        if not txt.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            emo, conf = predict_emotion_from_text(txt)
            st.success(f"😀 Detected emotion: **{emo}** (confidence {conf:.2f})")
            recs = recommend_for_emotion(emo, top_k=5)
            display_playable_recommendations(recs, emo)

with tab_cam:
    st.subheader("📸 Take a photo with your camera")
    cam_file = st.camera_input("Take a photo with your camera")
    if cam_file is not None:
        image = Image.open(cam_file).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        emo, conf = predict_emotion_from_face(bgr)
        if emo is None:
            st.warning("⚠️ Model not found. Please train the CNN first.")
        else:
            st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
            recs = recommend_for_emotion(emo, top_k=5)
            display_playable_recommendations(recs, emo)


