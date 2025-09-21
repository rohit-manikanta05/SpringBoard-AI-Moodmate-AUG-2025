# import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
# from PIL import Image
# import cv2
# import tensorflow as tf
# import webbrowser
# import urllib.parse
# import streamlit.components.v1 as components

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

# # --- Music Player Functions ---
# def create_youtube_search_url(title, artist):
#     """Create YouTube search URL for a song"""
#     query = f"{title} {artist}".strip()
#     encoded_query = urllib.parse.quote(query)
#     return f"https://www.youtube.com/results?search_query={encoded_query}"

# def create_spotify_search_url(title, artist):
#     """Create Spotify search URL for a song"""
#     query = f"{title} {artist}".strip()
#     encoded_query = urllib.parse.quote(query)
#     return f"https://open.spotify.com/search/{encoded_query}"

# def create_youtube_embed_player(song_query, height=300):
#     """Create an embedded YouTube player widget"""
#     # Note: This creates a search-based embed. For actual video IDs, you'd need YouTube API
#     search_query = urllib.parse.quote(song_query)
    
#     html_code = f'''
#     <div style="text-align: center; margin: 10px 0;">
#         <iframe width="100%" height="{height}" 
#                 src="https://www.youtube.com/embed?listType=search&list={search_query}&autoplay=0" 
#                 frameborder="0" 
#                 allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#                 allowfullscreen>
#         </iframe>
#     </div>
#     '''
#     return html_code

# def display_playable_recommendations(recs_df, emotion_name):
#     """Display recommendations with playback options"""
#     if recs_df.empty:
#         st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#         return
    
#     st.success(f"🎵 Here are your {emotion_name}-based music recommendations!")
    
#     # Add music service selection
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         music_service = st.selectbox(
#             "Choose your preferred music service:",
#             ["YouTube", "Spotify", "Embedded Player"],
#             key=f"service_{emotion_name}"
#         )
    
#     with col2:
#         auto_play_first = st.checkbox(
#             "Auto-play first recommendation", 
#             key=f"autoplay_{emotion_name}"
#         )
    
#     # Display recommendations with play buttons
#     for idx, row in recs_df.iterrows():
#         with st.container():
#             col1, col2, col3 = st.columns([3, 1, 1])
            
#             with col1:
#                 st.write(f"**{row['title']}** by {row['artist']}")
#                 st.caption(f"Genre: {row['genre']} | Mood: {row['mood']}")
#                 if 'score' in row:
#                     st.caption(f"Match Score: {row['score']:.3f}")
            
#             with col2:
#                 if music_service == "YouTube":
#                     youtube_url = create_youtube_search_url(row['title'], row['artist'])
#                     if st.button(f"▶️ Play", key=f"yt_{idx}"):
#                         webbrowser.open(youtube_url)
#                         st.success(f"Opening {row['title']} on YouTube...")
                
#                 elif music_service == "Spotify":
#                     spotify_url = create_spotify_search_url(row['title'], row['artist'])
#                     if st.button(f"▶️ Play", key=f"sp_{idx}"):
#                         webbrowser.open(spotify_url)
#                         st.success(f"Opening {row['title']} on Spotify...")
            
#             with col3:
#                 if st.button(f"📋 Copy", key=f"copy_{idx}"):
#                     song_info = f"{row['title']} - {row['artist']}"
#                     st.code(song_info)
#                     st.success("Song info copied!")
            
#             st.divider()
    
#     # Embedded player option
#     if music_service == "Embedded Player":
#         st.subheader("🎵 Embedded Music Player")
#         selected_song_idx = st.selectbox(
#             "Select a song to play:",
#             range(len(recs_df)),
#             format_func=lambda x: f"{recs_df.iloc[x]['title']} - {recs_df.iloc[x]['artist']}"
#         )
        
#         selected_song = recs_df.iloc[selected_song_idx]
#         song_query = f"{selected_song['title']} {selected_song['artist']}"
        
#         if st.button("🎵 Load Player", key="load_player"):
#             with st.spinner("Loading music player..."):
#                 player_html = create_youtube_embed_player(song_query)
#                 components.html(player_html, height=350)
    
#     # Auto-play first song if enabled
#     if auto_play_first and not recs_df.empty:
#         first_song = recs_df.iloc[0]
#         if music_service == "YouTube":
#             url = create_youtube_search_url(first_song['title'], first_song['artist'])
#         else:  # Spotify
#             url = create_spotify_search_url(first_song['title'], first_song['artist'])
        
#         st.info(f"🎵 Auto-playing: {first_song['title']} by {first_song['artist']}")
#         st.markdown(f"[▶️ Click here to play]({url})")

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
# st.title("🎵 MoodMate — Emotion → Music Playback")
# st.caption(
#     "Detect your emotion from a 📸 camera snapshot or 📂 uploaded photo, "
#     "or type a short text about how you feel to get mood-aligned songs that you can play instantly! 🎶"
# )


# # Add sidebar with music service info
# with st.sidebar:
#     st.header("🎵 Music Services")
#     st.info(
#         "**Available Options:**\n"
#         "• **YouTube**: Opens songs in YouTube\n"
#         "• **Spotify**: Opens songs in Spotify\n" 
#         "• **Embedded Player**: Play directly in the app"
#     )

#     st.warning(
#         "**Note**: For Spotify, you need to have the Spotify app installed or be logged into Spotify Web Player."
#     )

# tab_img, tab_txt, tab_cam, tab_chat = st.tabs(["📷 From Image", "✍️ From Text", "🎥 Real-time Webcam", "💬 Chatbot"])

# with tab_img:
#     st.subheader("📂 Upload a face photo")
    
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
#             recs = recommend_for_emotion(emo, top_k=5)
#             display_playable_recommendations(recs, emo)

# with tab_txt:
#     st.subheader("Describe how you feel")
#     txt = st.text_area("Type a sentence or two...", "")
#     if st.button("Analyze & Get Playable Playlist", type="primary"):
#         if not txt.strip():
#             st.warning("⚠️ Please enter some text.")
#         else:
#             emo, conf = predict_emotion_from_text(txt)
#             st.success(f"😀 Detected emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=5)
#             display_playable_recommendations(recs, emo)

# with tab_cam:
#     st.subheader("📸 Take a photo with your camera")
#     cam_file = st.camera_input("Take a photo with your camera")
#     if cam_file is not None:
#         image = Image.open(cam_file).convert("RGB")
#         bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         emo, conf = predict_emotion_from_face(bgr)
#         if emo is None:
#             st.warning("⚠️ Model not found. Please train the CNN first.")
#         else:
#             st.success(f"😀 Predicted emotion: **{emo}** (confidence {conf:.2f})")
#             recs = recommend_for_emotion(emo, top_k=5)
#             display_playable_recommendations(recs, emo)
# with tab_chat:
#   with tab_chat:
#     st.subheader("💬 Chat with MoodMate Bot")

#     # Initialize chat history in session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Simple rule-based chatbot function
#     def chatbot_with_recs(user_msg):
#         msg = user_msg.lower()

#         # Greeting
#         if any(g in msg for g in ["hi", "hello", "hey"]):
#             return "Hello! 😄 How are you feeling today?"

#         # Asking about emotions
#         elif any(e in msg for e in ["sad", "happy", "angry", "upset", "excited", "tired"]):
#             return f"Oh, you feel {msg}. I can recommend songs to match your mood! 🎵"

#         # Asking for music recommendations by mood
#         elif any(k in msg for k in ["song", "music", "recommend", "playlist", "play"]):
#             # Try to detect emotion keyword in message
#             found_emo = None
#             for emo in EMOTION_ID2NAME.values():
#                 if emo.lower() in msg:
#                     found_emo = emo.lower()
#                     break
            
#             if found_emo is None:
#                 # Default to 'happy' if no emotion detected
#                 found_emo = "happy"

#             recs_df = recommend_for_emotion(found_emo, top_k=3)
#             if recs_df.empty:
#                 return "Sorry, I don't have any songs to recommend right now. 😢"

#             # Build a string with song recommendations
#             songs_list = "\n".join([f"- **{row['title']}** by {row['artist']}" for _, row in recs_df.iterrows()])
#             return f"Here are some {found_emo}-mood songs for you:\n{songs_list}"

#         # Asking about app usage
#         elif any(k in msg for k in ["how", "use", "app", "function", "work"]):
#             return "You can upload a photo, take a selfie, or type how you feel. I’ll recommend songs based on your mood!"

#         # Thank you
#         elif any(k in msg for k in ["thanks", "thank you", "thx"]):
#             return "You're welcome! 😄 Enjoy your music!"

#         # Fallback
#         else:
#             return "I’m not sure about that. You can ask about moods, emotions, or music recommendations!"

#     # Input box for user message
#     user_input = st.text_area("Type your message here...", "")

#     if st.button("Send"):
#         if not user_input.strip():
#             st.warning("⚠️ Please type something to chat.")
#         else:
#             # Append user message
#             st.session_state.chat_history.append({"role": "user", "content": user_input})

#             # Get bot response
#             bot_reply = chatbot_with_recs(user_input)
#             st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

#     # Display chat history
#     for msg in st.session_state.chat_history:
#         if msg["role"] == "user":
#             st.markdown(f"**You:** {msg['content']}")
#         else:
#             st.markdown(f"**MoodMate Bot:** {msg['content']}")
#         st.divider()










# import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
# from PIL import Image
# import cv2
# import tensorflow as tf
# import webbrowser
# import urllib.parse
# import streamlit.components.v1 as components
# import requests
# import time
# import random
# from datetime import datetime, timedelta

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from src.utils.image import detect_and_crop_face
# from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

# st.set_page_config(
#     page_title="MoodMate Pro", 
#     page_icon="🎵",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

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

# # --- Enhanced CSS Styling ---
# def load_custom_css():
#     st.markdown("""
#     <style>
#     /* Main app styling */
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         text-align: center;
#         color: white;
#     }
    
#     /* Dynamic background based on mood */
#     .mood-happy { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
#     .mood-sad { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
#     .mood-angry { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
#     .mood-neutral { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
#     .mood-fear { background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); }
#     .mood-surprise { background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); }
#     .mood-disgust { background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%); }
    
#     /* Chat styling */
#     .chat-container {
#         max-height: 500px;
#         overflow-y: auto;
#         padding: 1rem;
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         background: #fafafa;
#     }
    
#     .user-message {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 10px 15px;
#         border-radius: 18px 18px 5px 18px;
#         margin: 10px 0;
#         max-width: 80%;
#         margin-left: auto;
#         display: block;
#     }
    
#     .bot-message {
#         background: #f1f3f4;
#         color: #333;
#         padding: 10px 15px;
#         border-radius: 18px 18px 18px 5px;
#         margin: 10px 0;
#         max-width: 80%;
#         border-left: 4px solid #667eea;
#     }
    
#     /* Music card styling */
#     .music-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#         margin: 10px 0;
#         border-left: 4px solid #667eea;
#         transition: transform 0.2s;
#     }
    
#     .music-card:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 20px rgba(0,0,0,0.15);
#     }
    
#     /* Animation classes */
#     .fade-in {
#         animation: fadeIn 0.5s ease-in;
#     }
    
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(20px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
    
#     /* Progress bar styling */
#     .stProgress > div > div > div > div {
#         background: linear-gradient(90deg, #667eea, #764ba2);
#     }
    
#     /* Sidebar styling */
#     .sidebar-content {
#         background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # --- Initialize Session State ---
# def init_session_state():
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "user_preferences" not in st.session_state:
#         st.session_state.user_preferences = {
#             "favorite_genres": [],
#             "favorite_artists": [],
#             "listening_history": [],
#             "current_mood": "neutral",
#             "music_service": "YouTube"
#         }
#     if "playlists" not in st.session_state:
#         st.session_state.playlists = {}
#     if "current_context" not in st.session_state:
#         st.session_state.current_context = "general"
#     if "quiz_score" not in st.session_state:
#         st.session_state.quiz_score = 0
#     if "quiz_questions" not in st.session_state:
#         st.session_state.quiz_questions = []

# # --- Enhanced Music Data ---
# MUSIC_TRIVIA = [
#     {"question": "Which artist has won the most Grammy Awards?", "answer": "Beyoncé", "options": ["Beyoncé", "Michael Jackson", "Taylor Swift", "Adele"]},
#     {"question": "What does 'BPM' stand for in music?", "answer": "Beats Per Minute", "options": ["Beats Per Minute", "Bass Per Measure", "Band Performance Metric", "Basic Pitch Modulation"]},
#     {"question": "Which instrument has 88 keys?", "answer": "Piano", "options": ["Piano", "Organ", "Harpsichord", "Synthesizer"]},
#     {"question": "What genre originated in New Orleans?", "answer": "Jazz", "options": ["Jazz", "Blues", "Country", "Rock"]},
#     {"question": "Who composed 'The Four Seasons'?", "answer": "Vivaldi", "options": ["Bach", "Mozart", "Vivaldi", "Beethoven"]}
# ]

# GENRES_INFO = {
#     "rock": {"color": "#FF6B6B", "description": "Energetic guitar-driven music", "mood": "energetic"},
#     "pop": {"color": "#4ECDC4", "description": "Popular mainstream music", "mood": "upbeat"},
#     "jazz": {"color": "#45B7D1", "description": "Improvisational and complex", "mood": "sophisticated"},
#     "classical": {"color": "#96CEB4", "description": "Orchestral and timeless", "mood": "elegant"},
#     "hip-hop": {"color": "#FFEAA7", "description": "Rhythmic and lyrical", "mood": "confident"},
#     "electronic": {"color": "#DDA0DD", "description": "Synthesized and digital", "mood": "futuristic"}
# }

# # --- Music Player Functions (Preserved) ---
# def create_youtube_search_url(title, artist):
#     """Create YouTube search URL for a song"""
#     query = f"{title} {artist}".strip()
#     encoded_query = urllib.parse.quote(query)
#     return f"https://www.youtube.com/results?search_query={encoded_query}"

# def create_spotify_search_url(title, artist):
#     """Create Spotify search URL for a song"""
#     query = f"{title} {artist}".strip()
#     encoded_query = urllib.parse.quote(query)
#     return f"https://open.spotify.com/search/{encoded_query}"

# def create_youtube_embed_player(song_query, height=300):
#     """Create an embedded YouTube player widget"""
#     search_query = urllib.parse.quote(song_query)
    
#     html_code = f'''
#     <div style="text-align: center; margin: 10px 0;">
#         <iframe width="100%" height="{height}" 
#                 src="https://www.youtube.com/embed?listType=search&list={search_query}&autoplay=0" 
#                 frameborder="0" 
#                 allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#                 allowfullscreen>
#         </iframe>
#     </div>
#     '''
#     return html_code

# # --- Enhanced Recommendations Display ---
# def display_enhanced_recommendations(recs_df, emotion_name):
#     """Enhanced display of recommendations with modern UI"""
#     if recs_df.empty:
#         st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
#         return
    
#     # Dynamic background based on emotion
#     mood_class = f"mood-{emotion_name.lower()}"
    
#     st.markdown(f"""
#     <div class="{mood_class} fade-in" style="padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
#         <h3 style="color: white; margin: 0;">🎵 {emotion_name.title()} Vibes Playlist</h3>
#         <p style="color: white; opacity: 0.9; margin: 0.5rem 0;">Curated just for your mood</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Music service selection with enhanced UI
#     col1, col2, col3 = st.columns([2, 1, 1])
#     with col1:
#         music_service = st.selectbox(
#             "🎵 Choose your music platform:",
#             ["YouTube", "Spotify", "Embedded Player"],
#             key=f"service_{emotion_name}",
#             index=["YouTube", "Spotify", "Embedded Player"].index(st.session_state.user_preferences["music_service"])
#         )
#         st.session_state.user_preferences["music_service"] = music_service
    
#     with col2:
#         create_playlist = st.button("💾 Create Playlist", key=f"create_playlist_{emotion_name}")
    
#     with col3:
#         export_playlist = st.button("📤 Export List", key=f"export_{emotion_name}")
    
#     # Display recommendations with enhanced cards
#     for idx, row in recs_df.iterrows():
#         with st.container():
#             st.markdown(f"""
#             <div class="music-card fade-in">
#                 <div style="display: flex; justify-content: space-between; align-items: center;">
#                     <div>
#                         <h4 style="margin: 0; color: #333;">🎵 {row['title']}</h4>
#                         <p style="margin: 0.5rem 0; color: #666;">👤 {row['artist']}</p>
#                         <p style="margin: 0; color: #888; font-size: 0.9rem;">
#                             🎭 {row['genre']} | 😊 {row['mood']}
#                             {f"| ⭐ {row['score']:.3f}" if 'score' in row else ""}
#                         </p>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Action buttons
#             col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
#             with col1:
#                 if st.button(f"▶️ Play", key=f"play_{idx}"):
#                     if music_service == "YouTube":
#                         url = create_youtube_search_url(row['title'], row['artist'])
#                         webbrowser.open(url)
#                         st.success(f"🎵 Opening on YouTube...")
#                     elif music_service == "Spotify":
#                         url = create_spotify_search_url(row['title'], row['artist'])
#                         webbrowser.open(url)
#                         st.success(f"🎵 Opening on Spotify...")
            
#             with col2:
#                 if st.button(f"❤️ Like", key=f"like_{idx}"):
#                     add_to_listening_history(row['title'], row['artist'], row['genre'])
#                     st.success("Added to favorites!")
            
#             with col3:
#                 if st.button(f"ℹ️ Info", key=f"info_{idx}"):
#                     show_song_info(row['title'], row['artist'], row['genre'])
            
#             with col4:
#                 if st.button(f"📋 Add to Playlist", key=f"add_playlist_{idx}"):
#                     add_to_temp_playlist(row)
    
#     # Playlist creation
#     if create_playlist:
#         create_mood_playlist(recs_df, emotion_name)
    
#     # Export functionality
#     if export_playlist:
#         export_playlist_data(recs_df, emotion_name)
    
#     # Embedded player
#     if music_service == "Embedded Player":
#         st.subheader("🎵 Embedded Music Player")
#         selected_idx = st.selectbox(
#             "Select a song:",
#             range(len(recs_df)),
#             format_func=lambda x: f"{recs_df.iloc[x]['title']} - {recs_df.iloc[x]['artist']}"
#         )
        
#         if st.button("🎵 Load Player", key="load_player"):
#             selected_song = recs_df.iloc[selected_idx]
#             song_query = f"{selected_song['title']} {selected_song['artist']}"
#             with st.spinner("🎵 Loading music player..."):
#                 player_html = create_youtube_embed_player(song_query)
#                 components.html(player_html, height=350)

# # --- Enhanced Helper Functions ---
# def add_to_listening_history(title, artist, genre):
#     """Add song to user's listening history"""
#     song_data = {
#         "title": title,
#         "artist": artist,
#         "genre": genre,
#         "timestamp": datetime.now().isoformat(),
#         "mood": st.session_state.user_preferences["current_mood"]
#     }
#     st.session_state.user_preferences["listening_history"].append(song_data)
    
#     # Update favorite genres
#     if genre not in st.session_state.user_preferences["favorite_genres"]:
#         st.session_state.user_preferences["favorite_genres"].append(genre)

# def show_song_info(title, artist, genre):
#     """Display detailed song information"""
#     st.info(f"""
#     **🎵 Song Details:**
#     - **Title**: {title}
#     - **Artist**: {artist}
#     - **Genre**: {genre}
#     - **Mood Match**: High
#     - **Similar Artists**: Based on your listening history
#     """)

# def add_to_temp_playlist(song_row):
#     """Add song to temporary playlist"""
#     if "temp_playlist" not in st.session_state:
#         st.session_state.temp_playlist = []
    
#     song_data = {
#         "title": song_row["title"],
#         "artist": song_row["artist"],
#         "genre": song_row["genre"],
#         "mood": song_row["mood"]
#     }
    
#     if song_data not in st.session_state.temp_playlist:
#         st.session_state.temp_playlist.append(song_data)
#         st.success(f"➕ Added '{song_row['title']}' to playlist!")

# def create_mood_playlist(recs_df, emotion_name):
#     """Create a named playlist from recommendations"""
#     playlist_name = f"{emotion_name.title()} Vibes - {datetime.now().strftime('%Y%m%d_%H%M')}"
#     playlist_data = []
    
#     for _, row in recs_df.iterrows():
#         playlist_data.append({
#             "title": row["title"],
#             "artist": row["artist"],
#             "genre": row["genre"],
#             "mood": row["mood"],
#             "added_date": datetime.now().isoformat()
#         })
    
#     st.session_state.playlists[playlist_name] = playlist_data
#     st.success(f"✅ Created playlist: '{playlist_name}' with {len(playlist_data)} songs!")

# def export_playlist_data(recs_df, emotion_name):
#     """Export playlist data"""
#     playlist_text = f"🎵 {emotion_name.title()} Mood Playlist\n"
#     playlist_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
#     for idx, row in recs_df.iterrows():
#         playlist_text += f"{idx+1}. {row['title']} - {row['artist']} ({row['genre']})\n"
    
#     st.download_button(
#         label="📥 Download Playlist",
#         data=playlist_text,
#         file_name=f"{emotion_name}_playlist_{datetime.now().strftime('%Y%m%d')}.txt",
#         mime="text/plain"
#     )

# # --- Music Quiz Functions ---
# def start_music_quiz():
#     """Start a music trivia quiz"""
#     st.session_state.quiz_questions = random.sample(MUSIC_TRIVIA, 3)
#     st.session_state.quiz_score = 0
#     st.session_state.current_question = 0

# def display_quiz_question():
#     """Display current quiz question"""
#     if st.session_state.quiz_questions:
#         q_idx = st.session_state.current_question
#         if q_idx < len(st.session_state.quiz_questions):
#             question = st.session_state.quiz_questions[q_idx]
            
#             st.markdown(f"**Question {q_idx + 1}:** {question['question']}")
            
#             answer = st.radio(
#                 "Choose your answer:",
#                 question['options'],
#                 key=f"quiz_q_{q_idx}"
#             )
            
#             if st.button("Submit Answer", key=f"submit_{q_idx}"):
#                 if answer == question['answer']:
#                     st.success("✅ Correct!")
#                     st.session_state.quiz_score += 1
#                 else:
#                     st.error(f"❌ Wrong! The correct answer is: {question['answer']}")
                
#                 st.session_state.current_question += 1
                
#                 if st.session_state.current_question >= len(st.session_state.quiz_questions):
#                     st.balloons()
#                     st.success(f"🎉 Quiz completed! Your score: {st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}")

# # --- Load Models (Preserved) ---
# MODEL_PATH = os.path.join("models", "fer_cnn.keras")
# CLASS_JSON = os.path.join("models", "class_names.json")

# cnn_model = None
# class_names = None
# if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
#     cnn_model = tf.keras.models.load_model(MODEL_PATH)
#     with open(CLASS_JSON) as f:
#         class_names = json.load(f)

# # Load recommender artifacts (Preserved)
# VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
# INDEX_PATH = os.path.join("models", "song_index.joblib")
# SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

# vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
# X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
# songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

# @st.cache_data
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

# # --- Enhanced Chatbot ---
# def enhanced_chatbot_response(user_msg):
#     """Enhanced chatbot with multiple contexts and features"""
#     msg = user_msg.lower()
    
#     # Context switching
#     if any(word in msg for word in ["quiz", "trivia", "game", "test"]):
#         st.session_state.current_context = "quiz"
#         return "🎮 Let's start a music quiz! I'll ask you some fun questions about music.", "quiz"
    
#     elif any(word in msg for word in ["playlist", "create playlist", "my playlists"]):
#         st.session_state.current_context = "playlist"
#         return "🎵 Let's work with playlists! You can create, view, or manage your playlists.", "playlist"
    
#     elif any(word in msg for word in ["search", "find song", "look for"]):
#         st.session_state.current_context = "search"
#         return "🔍 I can help you search for songs! Tell me what you're looking for.", "search"
    
#     elif any(word in msg for word in ["stats", "history", "listening"]):
#         st.session_state.current_context = "stats"
#         return "📊 Here are your music statistics and listening history!", "stats"
    
#     # Greeting responses
#     elif any(g in msg for g in ["hi", "hello", "hey", "what's up", "howdy"]):
#         greetings = [
#             "Hello! 🎵 Ready to discover some amazing music?",
#             "Hey there! 😄 What's your mood today?",
#             "Hi! 🎶 Let's find the perfect soundtrack for your day!",
#             "Hello! ✨ I'm here to help you with all things music!"
#         ]
#         return random.choice(greetings), "greeting"
    
#     # Emotion/mood responses
#     elif any(e in msg for e in ["sad", "happy", "angry", "upset", "excited", "tired", "energetic", "calm"]):
#         detected_emotions = []
#         for emotion in ["sad", "happy", "angry", "upset", "excited", "tired", "energetic", "calm"]:
#             if emotion in msg:
#                 detected_emotions.append(emotion)
        
#         if detected_emotions:
#             st.session_state.user_preferences["current_mood"] = detected_emotions[0]
#             return f"I can sense you're feeling {', '.join(detected_emotions)}. Let me recommend some perfect music for your mood! 🎵", "emotion"
    
#     # Music recommendation requests
#     elif any(k in msg for k in ["song", "music", "recommend", "playlist", "play", "listen"]):
#         # Try to detect emotion keyword in message
#         found_emo = None
#         for emo in EMOTION_ID2NAME.values():
#             if emo.lower() in msg:
#                 found_emo = emo.lower()
#                 break
        
#         if found_emo is None:
#             found_emo = st.session_state.user_preferences["current_mood"]
        
#         st.session_state.user_preferences["current_mood"] = found_emo
#         return f"🎵 Perfect! I'll find some great {found_emo} music for you. Check out the recommendations below!", "recommendation"
    
#     # Genre questions
#     elif any(genre in msg for genre in GENRES_INFO.keys()):
#         for genre, info in GENRES_INFO.items():
#             if genre in msg:
#                 return f"🎵 {genre.title()} music is {info['description']}. It typically creates a {info['mood']} mood. Would you like some {genre} recommendations?", "genre_info"
    
#     # Help requests
#     elif any(k in msg for k in ["how", "use", "app", "function", "work", "help", "what can you do"]):
#         return """
#         🎵 **I can help you with:**
#         • 📸 Analyze your emotions from photos
#         • 💭 Understand your mood from text
#         • 🎵 Recommend music based on your feelings
#         • 🎮 Play music trivia games
#         • 📊 Track your listening history
#         • 💾 Create and manage playlists
#         • 🔍 Search for songs and artists
#         • ℹ️ Get information about music and artists
        
#         Just tell me what you'd like to do!
#         """, "help"
    
#     # Thank you responses
#     elif any(k in msg for k in ["thanks", "thank you", "thx", "appreciate"]):
#         thanks_responses = [
#             "You're very welcome! 🎵 Keep enjoying the music!",
#             "Happy to help! 😄 Rock on! 🤘",
#             "My pleasure! 🎶 Let me know if you need more recommendations!",
#             "You're welcome! ✨ Music makes everything better!"
#         ]
#         return random.choice(thanks_responses), "thanks"
    
#     # Fallback with suggestions
#     else:
#         suggestions = [
#             "🤔 I'm not sure about that, but I can help you with music recommendations, mood analysis, or music trivia!",
#             "💭 Try asking me about your mood, music preferences, or say 'help' to see what I can do!",
#             "🎵 I specialize in music! You can ask about songs, emotions, create playlists, or play a music quiz!",
#             "✨ Let's talk about music! Tell me how you're feeling or what kind of music you like."
#         ]
#         return random.choice(suggestions), "fallback"

# # --- Main UI ---
# def main():
#     init_session_state()
#     load_custom_css()
    
#     # Enhanced header
#     st.markdown("""
#     <div class="main-header fade-in">
#         <h1>🎵 MoodMate Pro</h1>
#         <p>Advanced AI-Powered Music Companion</p>
#         <p>Emotion Detection • Smart Recommendations • Interactive Player • Music Trivia</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Enhanced sidebar
#     with st.sidebar:
#         st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
#         st.header("🎵 Your Music Profile")
        
#         # User stats
#         st.metric("🎧 Songs Played", len(st.session_state.user_preferences["listening_history"]))
#         st.metric("📋 Playlists Created", len(st.session_state.playlists))
#         st.metric("🎭 Current Mood", st.session_state.user_preferences["current_mood"].title())
        
#         # Favorite genres
#         if st.session_state.user_preferences["favorite_genres"]:
#             st.subheader("🎸 Favorite Genres")
#             for genre in st.session_state.user_preferences["favorite_genres"]:
#                 if genre in GENRES_INFO:
#                     color = GENRES_INFO[genre]["color"]
#                     st.markdown(f'<span style="background:{color}; color:white; padding:2px 8px; border-radius:12px; margin:2px;">{genre.title()}</span>', unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Quick actions
#         st.subheader("⚡ Quick Actions")
#         if st.button("🎮 Start Music Quiz"):
#             st.session_state.current_context = "quiz"
#             start_music_quiz()
        
#         if st.button("📊 View Statistics"):
#             st.session_state.current_context = "stats"
        
#         if st.button("💾 My Playlists"):
#             st.session_state.current_context = "playlist"
        
#         # Music service info
#         st.info("""
#         **🎵 Music Services:**
#         • **YouTube**: Opens in browser
#         • **Spotify**: Requires Spotify app
#         • **Embedded**: Play in app
#         """)

#     # Main tabs with enhanced functionality
#     tab_img, tab_txt, tab_cam, tab_chat, tab_features = st.tabs([
#         "📷 From Image", 
#         "✏️ From Text", 
#         "🎥 Real-time Webcam", 
#         "💬 Enhanced Chat", 
#         "🎵 Music Features"
#     ])

#     with tab_img:
#         st.subheader("📂 Upload a face photo")
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             img_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])
#         with col2:
#             if img_file:
#                 st.success("✅ Image loaded!")
        
#         if img_file is not None:
#             with st.spinner("🔍 Analyzing your emotion..."):
#                 progress = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress.progress(i + 1)
                
#                 image_bytes = img_file.read()
#                 image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#                 bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#                 col1, col2 = st.columns([1, 2])
#                 with col1:
#                     st.image(image, caption="Your Photo", width=200)
                
#                 with col2:
#                     emo, conf = predict_emotion_from_face(bgr)
#                     if emo is None:
#                         st.warning("⚠️ Model not found. Please train the CNN first.")
#                     else:
#                         st.session_state.user_preferences["current_mood"] = emo
#                         st.success(f"😀 Predicted emotion: **{emo.title()}** (confidence {conf:.2f})")
                        
#                         # Confidence gauge
#                         st.metric("🎯 Detection Confidence", f"{conf*100:.1f}%")
                        
#                         recs = recommend_for_emotion(emo, top_k=8)
#                         display_enhanced_recommendations(recs, emo)

#     with tab_txt:
#         st.subheader("💭 Describe how you feel")
        
#         # Quick mood buttons
#         st.write("**Quick Select:**")
#         mood_cols = st.columns(4)
#         quick_moods = ["😊 Happy", "😢 Sad", "😠 Angry", "😨 Anxious"]
        
#         for i, mood in enumerate(quick_moods):
#             with mood_cols[i]:
#                 if st.button(mood, key=f"quick_mood_{i}"):
#                     st.session_state.text_input = mood.split()[1].lower()
        
#         txt = st.text_area("Or type your feelings...", key="text_input", value=st.session_state.get("text_input", ""))
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             analyze_btn = st.button("🔍 Analyze & Get Playlist", type="primary")
#         with col2:
#             clear_btn = st.button("🗑️ Clear")
        
#         if clear_btn:
#             st.session_state.text_input = ""
#             st.experimental_rerun()
        
#         if analyze_btn:
#             if not txt.strip():
#                 st.warning("⚠️ Please enter some text.")
#             else:
#                 with st.spinner("🧠 Analyzing your emotions..."):
#                     progress = st.progress(0)
#                     for i in range(100):
#                         time.sleep(0.005)
#                         progress.progress(i + 1)
                    
#                     emo, conf = predict_emotion_from_text(txt)
#                     st.session_state.user_preferences["current_mood"] = emo
                    
#                     col1, col2 = st.columns([1, 1])
#                     with col1:
#                         st.success(f"😀 Detected emotion: **{emo.title()}**")
#                         st.metric("🎯 Confidence", f"{conf*100:.1f}%")
                    
#                     with col2:
#                         # Emotion visualization
#                         if emo in GENRES_INFO:
#                             color = GENRES_INFO.get(emo, {"color": "#667eea"})["color"]
#                             st.markdown(f"""
#                             <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center; color: white;">
#                                 <h3>Current Mood</h3>
#                                 <h2>{emo.title()}</h2>
#                             </div>
#                             """, unsafe_allow_html=True)
                    
#                     recs = recommend_for_emotion(emo, top_k=8)
#                     display_enhanced_recommendations(recs, emo)

#     with tab_cam:
#         st.subheader("📸 Take a photo with your camera")
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             cam_file = st.camera_input("Smile for the camera! 📸")
#         with col2:
#             if cam_file:
#                 st.success("📷 Photo captured!")
        
#         if cam_file is not None:
#             with st.spinner("📸 Processing your photo..."):
#                 image = Image.open(cam_file).convert("RGB")
#                 bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#                 emo, conf = predict_emotion_from_face(bgr)
#                 if emo is None:
#                     st.warning("⚠️ Model not found. Please train the CNN first.")
#                 else:
#                     st.session_state.user_preferences["current_mood"] = emo
#                     st.success(f"😀 Predicted emotion: **{emo.title()}** (confidence {conf:.2f})")
#                     recs = recommend_for_emotion(emo, top_k=8)
#                     display_enhanced_recommendations(recs, emo)

#     with tab_chat:
#         st.subheader("💬 Chat with Enhanced MoodMate Bot")
        
#         # Context indicator
#         context_colors = {
#             "general": "🔵", "quiz": "🎮", "playlist": "💾", 
#             "search": "🔍", "stats": "📊", "recommendation": "🎵"
#         }
#         current_context = st.session_state.current_context
#         st.info(f"{context_colors.get(current_context, '🔵')} Context: {current_context.title()}")
        
#         # Quick action buttons
#         st.write("**Quick Actions:**")
#         action_cols = st.columns(4)
#         quick_actions = [
#             ("🎵 Recommend Music", "recommend music for my mood"),
#             ("🎮 Start Quiz", "start music quiz"),
#             ("📊 My Stats", "show my listening statistics"),
#             ("💾 My Playlists", "show my playlists")
#         ]
        
#         for i, (label, action) in enumerate(quick_actions):
#             with action_cols[i]:
#                 if st.button(label, key=f"quick_action_{i}"):
#                     st.session_state.chat_history.append({"role": "user", "content": action})
#                     bot_reply, reply_context = enhanced_chatbot_response(action)
#                     st.session_state.chat_history.append({"role": "bot", "content": bot_reply, "context": reply_context})
        
#         # Chat interface
#         with st.container():
#             st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
#             # Display chat history
#             for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
#                 if msg["role"] == "user":
#                     st.markdown(f'<div class="user-message">👤 You: {msg["content"]}</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown(f'<div class="bot-message">🤖 MoodMate: {msg["content"]}</div>', unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         # Input area
#         col1, col2 = st.columns([4, 1])
#         with col1:
#             user_input = st.text_input("💬 Type your message...", key="chat_input")
#         with col2:
#             send_btn = st.button("📤 Send", key="send_chat")
        
#         # Handle specific contexts
#         if st.session_state.current_context == "quiz" and st.session_state.quiz_questions:
#             st.subheader("🎮 Music Quiz")
#             display_quiz_question()
        
#         elif st.session_state.current_context == "stats":
#             st.subheader("📊 Your Music Statistics")
            
#             if st.session_state.user_preferences["listening_history"]:
#                 history_df = pd.DataFrame(st.session_state.user_preferences["listening_history"])
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("🎵 Total Songs", len(history_df))
#                     st.metric("🎭 Genres Explored", len(history_df['genre'].unique()))
                
#                 with col2:
#                     # Most played genre
#                     top_genre = history_df['genre'].mode().iloc[0] if not history_df.empty else "None"
#                     st.metric("🏆 Top Genre", top_genre)
#                     st.metric("😊 Dominant Mood", history_df['mood'].mode().iloc[0] if not history_df.empty else "None")
                
#                 # Recent listening history
#                 st.subheader("🕒 Recent Plays")
#                 for _, song in history_df.tail(5).iterrows():
#                     st.write(f"🎵 **{song['title']}** by {song['artist']} - *{song['mood']} mood*")
#             else:
#                 st.info("🎵 Start listening to music to see your statistics!")
        
#         elif st.session_state.current_context == "playlist":
#             st.subheader("💾 Your Playlists")
            
#             if st.session_state.playlists:
#                 for playlist_name, songs in st.session_state.playlists.items():
#                     with st.expander(f"📋 {playlist_name} ({len(songs)} songs)"):
#                         for i, song in enumerate(songs, 1):
#                             st.write(f"{i}. **{song['title']}** by {song['artist']}")
#             else:
#                 st.info("💾 No playlists created yet. Create one from your recommendations!")
        
#         # Process chat input
#         if send_btn and user_input.strip():
#             # Add user message
#             st.session_state.chat_history.append({"role": "user", "content": user_input})
            
#             # Get bot response
#             bot_reply, reply_context = enhanced_chatbot_response(user_input)
#             st.session_state.chat_history.append({"role": "bot", "content": bot_reply, "context": reply_context})
            
#             # Handle music recommendation context
#             if reply_context == "recommendation":
#                 current_mood = st.session_state.user_preferences["current_mood"]
#                 recs = recommend_for_emotion(current_mood, top_k=5)
#                 if not recs.empty:
#                     display_enhanced_recommendations(recs, current_mood)
            
#             # Clear input and rerun
#             st.session_state.chat_input = ""
#             st.experimental_rerun()

#     with tab_features:
#         st.subheader("🎵 Advanced Music Features")
        
#         feature_tabs = st.tabs(["🔍 Music Search", "🎲 Music Discovery", "📈 Trends", "🎪 Fun Features"])
        
#         with feature_tabs[0]:  # Music Search
#             st.subheader("🔍 Advanced Music Search")
            
#             search_type = st.selectbox("Search by:", ["Song Title", "Artist", "Genre", "Mood"])
#             search_query = st.text_input(f"Search for {search_type.lower()}...")
            
#             if st.button("🔍 Search") and search_query:
#                 with st.spinner("🔍 Searching..."):
#                     # Simulate search results
#                     st.success(f"Found results for '{search_query}'")
                    
#                     # Mock search results
#                     search_results = [
#                         {"title": f"Song matching '{search_query}'", "artist": "Various Artists", "genre": "Pop", "mood": "happy"},
#                         {"title": f"Another match for '{search_query}'", "artist": "Demo Artist", "genre": "Rock", "mood": "energetic"}
#                     ]
                    
#                     for result in search_results:
#                         st.markdown(f"""
#                         <div class="music-card">
#                             <h4>🎵 {result['title']}</h4>
#                             <p>👤 {result['artist']} | 🎭 {result['genre']} | 😊 {result['mood']}</p>
#                         </div>
#                         """, unsafe_allow_html=True)
        
#         with feature_tabs[1]:  # Music Discovery
#             st.subheader("🎲 Music Discovery")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("🎲 Random Song Discovery"):
#                     if songs_df is not None and not songs_df.empty:
#                         random_song = songs_df.sample(1).iloc[0]
#                         st.success("🎵 Here's a random discovery!")
#                         st.markdown(f"""
#                         <div class="music-card fade-in">
#                             <h4>🎵 {random_song['title']}</h4>
#                             <p>👤 {random_song['artist']} | 🎭 {random_song['genre']}</p>
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             with col2:
#                 if st.button("🌟 Mood-based Discovery"):
#                     current_mood = st.session_state.user_preferences["current_mood"]
#                     discovery_recs = recommend_for_emotion(current_mood, top_k=3)
#                     if not discovery_recs.empty:
#                         st.success(f"🌟 Discoveries for {current_mood} mood!")
#                         for _, song in discovery_recs.iterrows():
#                             st.markdown(f"🎵 **{song['title']}** by {song['artist']}")
        
#         with feature_tabs[2]:  # Trends
#             st.subheader("📈 Music Trends & Analytics")
            
#             if st.session_state.user_preferences["listening_history"]:
#                 history_df = pd.DataFrame(st.session_state.user_preferences["listening_history"])
                
#                 # Genre distribution
#                 genre_counts = history_df['genre'].value_counts()
#                 st.subheader("🎭 Your Genre Preferences")
                
#                 for genre, count in genre_counts.items():
#                     percentage = (count / len(history_df)) * 100
#                     st.progress(percentage / 100)
#                     st.write(f"**{genre.title()}**: {count} plays ({percentage:.1f}%)")
                
#                 # Mood trends
#                 st.subheader("😊 Mood Trends")
#                 mood_counts = history_df['mood'].value_counts()
#                 for mood, count in mood_counts.head(3).items():
#                     st.write(f"**{mood.title()}**: {count} times")
#             else:
#                 st.info("📊 Listen to more music to see your trends!")
        
#         with feature_tabs[3]:  # Fun Features
#             st.subheader("🎪 Fun Music Features")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎵 Song of the Day")
#                 if st.button("🌅 Get Today's Song"):
#                     if songs_df is not None and not songs_df.empty:
#                         # Use date as seed for consistent daily song
#                         today = datetime.now().date()
#                         np.random.seed(hash(str(today)) % 2**32)
#                         daily_song = songs_df.sample(1).iloc[0]
                        
#                         st.success("🌅 Your song for today!")
#                         st.markdown(f"""
#                         <div class="music-card fade-in" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
#                             <h4>🎵 {daily_song['title']}</h4>
#                             <p>👤 {daily_song['artist']} | 🎭 {daily_song['genre']}</p>
#                             <p><em>Perfect for {today.strftime('%A, %B %d')}</em></p>
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             with col2:
#                 st.subheader("🎯 Music Mood Match")
#                 if st.button("🎯 Find My Perfect Match"):
#                     current_mood = st.session_state.user_preferences["current_mood"]
#                     perfect_match = recommend_for_emotion(current_mood, top_k=1)
                    
#                     if not perfect_match.empty:
#                         match = perfect_match.iloc[0]
#                         st.success("🎯 Perfect match found!")
#                         st.markdown(f"""
#                         <div class="music-card fade-in" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
#                             <h4>💯 {match['title']}</h4>
#                             <p>👤 {match['artist']}</p>
#                             <p>🎯 Match Score: {match.get('score', 0):.3f}</p>
#                         </div>
#                         """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()












import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
import webbrowser
import urllib.parse
import streamlit.components.v1 as components
import requests
import time
import random
from datetime import datetime, timedelta

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.image import detect_and_crop_face
from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

st.set_page_config(
    page_title="MoodMate Pro", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Enhanced CSS Styling ---
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Dynamic background based on mood */
    .mood-happy { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .mood-sad { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .mood-angry { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .mood-neutral { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
    .mood-fear { background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); }
    .mood-surprise { background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); }
    .mood-disgust { background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%); }
    
    /* Chat styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        background: #fafafa;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        display: block;
    }
    
    .bot-message {
        background: #f1f3f4;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #667eea;
    }
    
    /* Music card styling */
    .music-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    .music-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "favorite_genres": [],
            "favorite_artists": [],
            "listening_history": [],
            "current_mood": "neutral",
            "music_service": "YouTube"
        }
    if "playlists" not in st.session_state:
        st.session_state.playlists = {}
    if "current_context" not in st.session_state:
        st.session_state.current_context = "general"
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False
    if "recommendation_emotion" not in st.session_state:
        st.session_state.recommendation_emotion = None

# --- Enhanced Music Data ---
MUSIC_TRIVIA = [
    {"question": "Which artist has won the most Grammy Awards?", "answer": "Beyoncé", "options": ["Beyoncé", "Michael Jackson", "Taylor Swift", "Adele"]},
    {"question": "What does 'BPM' stand for in music?", "answer": "Beats Per Minute", "options": ["Beats Per Minute", "Bass Per Measure", "Band Performance Metric", "Basic Pitch Modulation"]},
    {"question": "Which instrument has 88 keys?", "answer": "Piano", "options": ["Piano", "Organ", "Harpsichord", "Synthesizer"]},
    {"question": "What genre originated in New Orleans?", "answer": "Jazz", "options": ["Jazz", "Blues", "Country", "Rock"]},
    {"question": "Who composed 'The Four Seasons'?", "answer": "Vivaldi", "options": ["Bach", "Mozart", "Vivaldi", "Beethoven"]}
]

GENRES_INFO = {
    "rock": {"color": "#FF6B6B", "description": "Energetic guitar-driven music", "mood": "energetic"},
    "pop": {"color": "#4ECDC4", "description": "Popular mainstream music", "mood": "upbeat"},
    "jazz": {"color": "#45B7D1", "description": "Improvisational and complex", "mood": "sophisticated"},
    "classical": {"color": "#96CEB4", "description": "Orchestral and timeless", "mood": "elegant"},
    "hip-hop": {"color": "#FFEAA7", "description": "Rhythmic and lyrical", "mood": "confident"},
    "electronic": {"color": "#DDA0DD", "description": "Synthesized and digital", "mood": "futuristic"}
}

# --- Music Player Functions (Preserved) ---
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

# --- Enhanced Recommendations Display ---
def display_enhanced_recommendations(recs_df, emotion_name):
    """Enhanced display of recommendations with modern UI"""
    if recs_df.empty:
        st.info("ℹ️ Recommender index missing. Run the recommender builder script.")
        return
    
    # Dynamic background based on emotion
    mood_class = f"mood-{emotion_name.lower()}"
    
    st.markdown(f"""
    <div class="{mood_class} fade-in" style="padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🎵 {emotion_name.title()} Vibes Playlist</h3>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0;">Curated just for your mood</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Music service selection with enhanced UI
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        music_service = st.selectbox(
            "🎵 Choose your music platform:",
            ["YouTube", "Spotify", "Embedded Player"],
            key=f"service_{emotion_name}",
            index=["YouTube", "Spotify", "Embedded Player"].index(st.session_state.user_preferences["music_service"])
        )
        st.session_state.user_preferences["music_service"] = music_service
    
    with col2:
        create_playlist = st.button("💾 Create Playlist", key=f"create_playlist_{emotion_name}")
    
    with col3:
        export_playlist = st.button("📤 Export List", key=f"export_{emotion_name}")
    
    # Display recommendations with enhanced cards
    for idx, row in recs_df.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="music-card fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #333;">🎵 {row['title']}</h4>
                        <p style="margin: 0.5rem 0; color: #666;">👤 {row['artist']}</p>
                        <p style="margin: 0; color: #888; font-size: 0.9rem;">
                            🎭 {row['genre']} | 😊 {row['mood']}
                            {f"| ⭐ {row['score']:.3f}" if 'score' in row else ""}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if st.button(f"▶️ Play", key=f"play_{idx}"):
                    if music_service == "YouTube":
                        url = create_youtube_search_url(row['title'], row['artist'])
                        webbrowser.open(url)
                        st.success(f"🎵 Opening on YouTube...")
                    elif music_service == "Spotify":
                        url = create_spotify_search_url(row['title'], row['artist'])
                        webbrowser.open(url)
                        st.success(f"🎵 Opening on Spotify...")
            
            with col2:
                if st.button(f"❤️ Like", key=f"like_{idx}"):
                    add_to_listening_history(row['title'], row['artist'], row['genre'])
                    st.success("Added to favorites!")
            
            with col3:
                if st.button(f"ℹ️ Info", key=f"info_{idx}"):
                    show_song_info(row['title'], row['artist'], row['genre'])
            
            with col4:
                if st.button(f"📋 Add to Playlist", key=f"add_playlist_{idx}"):
                    add_to_temp_playlist(row)
    
    # Playlist creation
    if create_playlist:
        create_mood_playlist(recs_df, emotion_name)
    
    # Export functionality
    if export_playlist:
        export_playlist_data(recs_df, emotion_name)
    
    # Embedded player
    if music_service == "Embedded Player":
        st.subheader("🎵 Embedded Music Player")
        selected_idx = st.selectbox(
            "Select a song:",
            range(len(recs_df)),
            format_func=lambda x: f"{recs_df.iloc[x]['title']} - {recs_df.iloc[x]['artist']}"
        )
        
        if st.button("🎵 Load Player", key="load_player"):
            selected_song = recs_df.iloc[selected_idx]
            song_query = f"{selected_song['title']} {selected_song['artist']}"
            with st.spinner("🎵 Loading music player..."):
                player_html = create_youtube_embed_player(song_query)
                components.html(player_html, height=350)

# --- Enhanced Helper Functions ---
def add_to_listening_history(title, artist, genre):
    """Add song to user's listening history"""
    song_data = {
        "title": title,
        "artist": artist,
        "genre": genre,
        "timestamp": datetime.now().isoformat(),
        "mood": st.session_state.user_preferences["current_mood"]
    }
    st.session_state.user_preferences["listening_history"].append(song_data)
    
    # Update favorite genres
    if genre not in st.session_state.user_preferences["favorite_genres"]:
        st.session_state.user_preferences["favorite_genres"].append(genre)

def show_song_info(title, artist, genre):
    """Display detailed song information"""
    st.info(f"""
    **🎵 Song Details:**
    - **Title**: {title}
    - **Artist**: {artist}
    - **Genre**: {genre}
    - **Mood Match**: High
    - **Similar Artists**: Based on your listening history
    """)

def add_to_temp_playlist(song_row):
    """Add song to temporary playlist"""
    if "temp_playlist" not in st.session_state:
        st.session_state.temp_playlist = []
    
    song_data = {
        "title": song_row["title"],
        "artist": song_row["artist"],
        "genre": song_row["genre"],
        "mood": song_row["mood"]
    }
    
    if song_data not in st.session_state.temp_playlist:
        st.session_state.temp_playlist.append(song_data)
        st.success(f"➕ Added '{song_row['title']}' to playlist!")

def create_mood_playlist(recs_df, emotion_name):
    """Create a named playlist from recommendations"""
    playlist_name = f"{emotion_name.title()} Vibes - {datetime.now().strftime('%Y%m%d_%H%M')}"
    playlist_data = []
    
    for _, row in recs_df.iterrows():
        playlist_data.append({
            "title": row["title"],
            "artist": row["artist"],
            "genre": row["genre"],
            "mood": row["mood"],
            "added_date": datetime.now().isoformat()
        })
    
    st.session_state.playlists[playlist_name] = playlist_data
    st.success(f"✅ Created playlist: '{playlist_name}' with {len(playlist_data)} songs!")

def export_playlist_data(recs_df, emotion_name):
    """Export playlist data"""
    playlist_text = f"🎵 {emotion_name.title()} Mood Playlist\n"
    playlist_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    for idx, row in recs_df.iterrows():
        playlist_text += f"{idx+1}. {row['title']} - {row['artist']} ({row['genre']})\n"
    
    st.download_button(
        label="📥 Download Playlist",
        data=playlist_text,
        file_name=f"{emotion_name}_playlist_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# --- Music Quiz Functions ---
def start_music_quiz():
    """Start a music trivia quiz"""
    st.session_state.quiz_questions = random.sample(MUSIC_TRIVIA, 3)
    st.session_state.quiz_score = 0
    st.session_state.current_question = 0

def display_quiz_question():
    """Display current quiz question"""
    if st.session_state.quiz_questions:
        q_idx = st.session_state.current_question
        if q_idx < len(st.session_state.quiz_questions):
            question = st.session_state.quiz_questions[q_idx]
            
            st.markdown(f"**Question {q_idx + 1}:** {question['question']}")
            
            answer = st.radio(
                "Choose your answer:",
                question['options'],
                key=f"quiz_q_{q_idx}"
            )
            
            if st.button("Submit Answer", key=f"submit_{q_idx}"):
                if answer == question['answer']:
                    st.success("✅ Correct!")
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"❌ Wrong! The correct answer is: {question['answer']}")
                
                st.session_state.current_question += 1
                
                if st.session_state.current_question >= len(st.session_state.quiz_questions):
                    st.balloons()
                    st.success(f"🎉 Quiz completed! Your score: {st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}")

# --- Load Models (Preserved) ---
MODEL_PATH = os.path.join("models", "fer_cnn.keras")
CLASS_JSON = os.path.join("models", "class_names.json")

cnn_model = None
class_names = None
if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON) as f:
        class_names = json.load(f)

# Load recommender artifacts (Preserved)
VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
INDEX_PATH = os.path.join("models", "song_index.joblib")
SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

@st.cache_data
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

    for emo, kw in cues.items():
        if any(k in text_l for k in kw):
            return emo, 0.9

    if compound >= 0.5:
        return "happy", compound
    elif compound <= -0.6:
        if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
            return "angry", abs(compound)
        return "sad", abs(compound)
    else:
        return "neutral", 1.0 - abs(compound)

# --- Enhanced Chatbot ---
def enhanced_chatbot_response(user_msg):
    """Enhanced chatbot with multiple contexts and features"""
    msg = user_msg.lower()
    
    # Context switching
    if any(word in msg for word in ["quiz", "trivia", "game", "test"]):
        st.session_state.current_context = "quiz"
        return "🎮 Let's start a music quiz! I'll ask you some fun questions about music.", "quiz"
    
    elif any(word in msg for word in ["playlist", "create playlist", "my playlists"]):
        st.session_state.current_context = "playlist"
        return "🎵 Let's work with playlists! You can create, view, or manage your playlists.", "playlist"
    
    elif any(word in msg for word in ["search", "find song", "look for"]):
        st.session_state.current_context = "search"
        return "🔍 I can help you search for songs! Tell me what you're looking for.", "search"
    
    elif any(word in msg for word in ["stats", "history", "listening"]):
        st.session_state.current_context = "stats"
        return "📊 Here are your music statistics and listening history!", "stats"
    
    # Greeting responses
    elif any(g in msg for g in ["hi", "hello", "hey", "what's up", "howdy"]):
        greetings = [
            "Hello! 🎵 Ready to discover some amazing music?",
            "Hey there! 😄 What's your mood today?",
            "Hi! 🎶 Let's find the perfect soundtrack for your day!",
            "Hello! ✨ I'm here to help you with all things music!"
        ]
        return random.choice(greetings), "greeting"
    
    # Emotion/mood responses
    elif any(e in msg for e in ["sad", "happy", "angry", "upset", "excited", "tired", "energetic", "calm"]):
        detected_emotions = []
        for emotion in ["sad", "happy", "angry", "upset", "excited", "tired", "energetic", "calm"]:
            if emotion in msg:
                detected_emotions.append(emotion)
        
        if detected_emotions:
            st.session_state.user_preferences["current_mood"] = detected_emotions[0]
            return f"I can sense you're feeling {', '.join(detected_emotions)}. Let me recommend some perfect music for your mood! 🎵", "emotion"
    
    # Music recommendation requests - FIXED VERSION
    elif any(k in msg for k in ["song", "music", "recommend", "playlist", "play", "listen"]):
        # Try to detect emotion keyword in message
        found_emo = None
        for emo in EMOTION_ID2NAME.values():
            if emo.lower() in msg:
                found_emo = emo.lower()
                break
        
        if found_emo is None:
            found_emo = st.session_state.user_preferences["current_mood"]
        
        st.session_state.user_preferences["current_mood"] = found_emo
        
        # Set flags to trigger immediate recommendations
        st.session_state.show_recommendations = True
        st.session_state.recommendation_emotion = found_emo
        
        return f"🎵 Perfect! I'll find some great {found_emo} music for you. Check out the recommendations below!", "recommendation"
    
    # Genre questions
    elif any(genre in msg for genre in GENRES_INFO.keys()):
        for genre, info in GENRES_INFO.items():
            if genre in msg:
                return f"🎵 {genre.title()} music is {info['description']}. It typically creates a {info['mood']} mood. Would you like some {genre} recommendations?", "genre_info"
    
    # Help requests
    elif any(k in msg for k in ["how", "use", "app", "function", "work", "help", "what can you do"]):
        return """
        🎵 **I can help you with:**
        • 📸 Analyze your emotions from photos
        • 💭 Understand your mood from text
        • 🎵 Recommend music based on your feelings
        • 🎮 Play music trivia games
        • 📊 Track your listening history
        • 💾 Create and manage playlists
        • 🔍 Search for songs and artists
        • ℹ️ Get information about music and artists
        
        Just tell me what you'd like to do!
        """, "help"
    
    # Thank you responses
    elif any(k in msg for k in ["thanks", "thank you", "thx", "appreciate"]):
        thanks_responses = [
            "You're very welcome! 🎵 Keep enjoying the music!",
            "Happy to help! 😄 Rock on! 🤘",
            "My pleasure! 🎶 Let me know if you need more recommendations!",
            "You're welcome! ✨ Music makes everything better!"
        ]
        return random.choice(thanks_responses), "thanks"
    
    # Fallback with suggestions
    else:
        suggestions = [
            "🤔 I'm not sure about that, but I can help you with music recommendations, mood analysis, or music trivia!",
            "💭 Try asking me about your mood, music preferences, or say 'help' to see what I can do!",
            "🎵 I specialize in music! You can ask about songs, emotions, create playlists, or play a music quiz!",
            "✨ Let's talk about music! Tell me how you're feeling or what kind of music you like."
        ]
        return random.choice(suggestions), "fallback"

# --- Main UI ---
def main():
    init_session_state()
    load_custom_css()
    
    # Enhanced header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🎵 MoodMate Pro</h1>
        <p>Advanced AI-Powered Music Companion</p>
        <p>Emotion Detection • Smart Recommendations • Interactive Player • Music Trivia</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("🎵 Your Music Profile")
        
        # User stats
        st.metric("🎧 Songs Played", len(st.session_state.user_preferences["listening_history"]))
        st.metric("📋 Playlists Created", len(st.session_state.playlists))
        st.metric("🎭 Current Mood", st.session_state.user_preferences["current_mood"].title())
        
        # Favorite genres
        if st.session_state.user_preferences["favorite_genres"]:
            st.subheader("🎸 Favorite Genres")
            for genre in st.session_state.user_preferences["favorite_genres"]:
                if genre in GENRES_INFO:
                    color = GENRES_INFO[genre]["color"]
                    st.markdown(f'<span style="background:{color}; color:white; padding:2px 8px; border-radius:12px; margin:2px;">{genre.title()}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🎮 Start Music Quiz"):
            st.session_state.current_context = "quiz"
            start_music_quiz()
        
        if st.button("📊 View Statistics"):
            st.session_state.current_context = "stats"
        
        if st.button("💾 My Playlists"):
            st.session_state.current_context = "playlist"
        
        # Music service info
        st.info("""
        **🎵 Music Services:**
        • **YouTube**: Opens in browser
        • **Spotify**: Requires Spotify app
        • **Embedded**: Play in app
        """)

    # Main tabs with enhanced functionality
    tab_img, tab_txt, tab_cam, tab_chat, tab_features = st.tabs([
        "📷 From Image", 
        "✏️ From Text", 
        "🎥 Real-time Webcam", 
        "💬 Enhanced Chat", 
        "🎵 Music Features"
    ])

    with tab_img:
        st.subheader("📂 Upload a face photo")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            img_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])
        with col2:
            if img_file:
                st.success("✅ Image loaded!")
        
        if img_file is not None:
            with st.spinner("🔍 Analyzing your emotion..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                
                image_bytes = img_file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="Your Photo", width=200)
                
                with col2:
                    emo, conf = predict_emotion_from_face(bgr)
                    if emo is None:
                        st.warning("⚠️ Model not found. Please train the CNN first.")
                    else:
                        st.session_state.user_preferences["current_mood"] = emo
                        st.success(f"😀 Predicted emotion: **{emo.title()}** (confidence {conf:.2f})")
                        
                        # Confidence gauge
                        st.metric("🎯 Detection Confidence", f"{conf*100:.1f}%")
                        
                        recs = recommend_for_emotion(emo, top_k=8)
                        display_enhanced_recommendations(recs, emo)

    with tab_txt:
        st.subheader("💭 Describe how you feel")
        
        # Quick mood buttons
        st.write("**Quick Select:**")
        mood_cols = st.columns(4)
        quick_moods = ["😊 Happy", "😢 Sad", "😠 Angry", "😨 Anxious"]
        selected_mood = None
        
        for i, mood in enumerate(quick_moods):
            with mood_cols[i]:
                if st.button(mood, key=f"quick_mood_{i}"):
                    selected_mood = mood.split()[1].lower()
        
        # Use selected mood or manual input
        if selected_mood:
            txt = selected_mood
        else:
            txt = st.text_area("Or type your feelings...", key="text_input_area")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            analyze_btn = st.button("🔍 Analyze & Get Playlist", type="primary")
        with col2:
            clear_btn = st.button("🗑️ Clear")
        
        if clear_btn:
            st.rerun()
        
        if analyze_btn:
            if not txt.strip():
                st.warning("⚠️ Please enter some text.")
            else:
                with st.spinner("🧠 Analyzing your emotions..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.005)
                        progress.progress(i + 1)
                    
                    emo, conf = predict_emotion_from_text(txt)
                    st.session_state.user_preferences["current_mood"] = emo
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.success(f"😀 Detected emotion: **{emo.title()}**")
                        st.metric("🎯 Confidence", f"{conf*100:.1f}%")
                    
                    with col2:
                        # Emotion visualization
                        if emo in GENRES_INFO:
                            color = GENRES_INFO.get(emo, {"color": "#667eea"})["color"]
                            st.markdown(f"""
                            <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                                <h3>Current Mood</h3>
                                <h2>{emo.title()}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    recs = recommend_for_emotion(emo, top_k=8)
                    display_enhanced_recommendations(recs, emo)

    with tab_cam:
        st.subheader("📸 Take a photo with your camera")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            cam_file = st.camera_input("Smile for the camera! 📸")
        with col2:
            if cam_file:
                st.success("📷 Photo captured!")
        
        if cam_file is not None:
            with st.spinner("📸 Processing your photo..."):
                image = Image.open(cam_file).convert("RGB")
                bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                emo, conf = predict_emotion_from_face(bgr)
                if emo is None:
                    st.warning("⚠️ Model not found. Please train the CNN first.")
                else:
                    st.session_state.user_preferences["current_mood"] = emo
                    st.success(f"😀 Predicted emotion: **{emo.title()}** (confidence {conf:.2f})")
                    recs = recommend_for_emotion(emo, top_k=8)
                    display_enhanced_recommendations(recs, emo)

    with tab_chat:
        st.subheader("💬 Chat with Enhanced MoodMate Bot")
        
        # Context indicator
        context_colors = {
            "general": "🔵", "quiz": "🎮", "playlist": "💾", 
            "search": "🔍", "stats": "📊", "recommendation": "🎵"
        }
        current_context = st.session_state.current_context
        st.info(f"{context_colors.get(current_context, '🔵')} Context: {current_context.title()}")
        
        # Quick action buttons
        st.write("**Quick Actions:**")
        action_cols = st.columns(4)
        quick_actions = [
            ("🎵 Recommend Music", "recommend music for my mood"),
            ("🎮 Start Quiz", "start music quiz"),
            ("📊 My Stats", "show my listening statistics"),
            ("💾 My Playlists", "show my playlists")
        ]
        
        for i, (label, action) in enumerate(quick_actions):
            with action_cols[i]:
                if st.button(label, key=f"quick_action_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": action})
                    bot_reply, reply_context = enhanced_chatbot_response(action)
                    st.session_state.chat_history.append({"role": "bot", "content": bot_reply, "context": reply_context})
        
        # Chat interface
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history
            for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 You: {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 MoodMate: {msg["content"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("💬 Type your message...", key="chat_input", value="")
        with col2:
            send_btn = st.button("📤 Send", key="send_chat")
        
        # Handle specific contexts
        if st.session_state.current_context == "quiz" and st.session_state.quiz_questions:
            st.subheader("🎮 Music Quiz")
            display_quiz_question()
        
        elif st.session_state.current_context == "stats":
            st.subheader("📊 Your Music Statistics")
            
            if st.session_state.user_preferences["listening_history"]:
                history_df = pd.DataFrame(st.session_state.user_preferences["listening_history"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎵 Total Songs", len(history_df))
                    st.metric("🎭 Genres Explored", len(history_df['genre'].unique()))
                
                with col2:
                    # Most played genre
                    top_genre = history_df['genre'].mode().iloc[0] if not history_df.empty else "None"
                    st.metric("🏆 Top Genre", top_genre)
                    st.metric("😊 Dominant Mood", history_df['mood'].mode().iloc[0] if not history_df.empty else "None")
                
                # Recent listening history
                st.subheader("🕒 Recent Plays")
                for _, song in history_df.tail(5).iterrows():
                    st.write(f"🎵 **{song['title']}** by {song['artist']} - *{song['mood']} mood*")
            else:
                st.info("🎵 Start listening to music to see your statistics!")
        
        elif st.session_state.current_context == "playlist":
            st.subheader("💾 Your Playlists")
            
            if st.session_state.playlists:
                for playlist_name, songs in st.session_state.playlists.items():
                    with st.expander(f"📋 {playlist_name} ({len(songs)} songs)"):
                        for i, song in enumerate(songs, 1):
                            st.write(f"{i}. **{song['title']}** by {song['artist']}")
            else:
                st.info("💾 No playlists created yet. Create one from your recommendations!")
        
        # Process chat input - FIXED VERSION
        if send_btn and user_input.strip():
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            bot_reply, reply_context = enhanced_chatbot_response(user_input)
            st.session_state.chat_history.append({"role": "bot", "content": bot_reply, "context": reply_context})
            
            # Rerun to clear the input and show new messages
            st.rerun()
        
        # Check if recommendations should be displayed - NEW SECTION
        if st.session_state.show_recommendations and st.session_state.recommendation_emotion:
            emotion = st.session_state.recommendation_emotion
            st.subheader(f"🎵 Your {emotion.title()} Music Recommendations")
            
            with st.spinner("🎵 Finding perfect music for your mood..."):
                recs = recommend_for_emotion(emotion, top_k=8)
                
                if not recs.empty:
                    display_enhanced_recommendations(recs, emotion)
                else:
                    st.error("❌ Sorry, I couldn't find music recommendations. The music database might not be loaded properly.")
                    st.info("💡 Make sure the following files exist:\n- models/tfidf_vectorizer.joblib\n- models/song_index.joblib\n- models/songs_clean.parquet")
            
            # Clear the recommendation flags
            st.session_state.show_recommendations = False
            st.session_state.recommendation_emotion = None

    with tab_features:
        st.subheader("🎵 Advanced Music Features")
        
        feature_tabs = st.tabs(["🔍 Music Search", "🎲 Music Discovery", "📈 Trends", "🎪 Fun Features"])
        
        with feature_tabs[0]:  # Music Search
            st.subheader("🔍 Advanced Music Search")
            
            search_type = st.selectbox("Search by:", ["Song Title", "Artist", "Genre", "Mood"])
            search_query = st.text_input(f"Search for {search_type.lower()}...")
            
            if st.button("🔍 Search") and search_query:
                with st.spinner("🔍 Searching..."):
                    # Simulate search results
                    st.success(f"Found results for '{search_query}'")
                    
                    # Mock search results
                    search_results = [
                        {"title": f"Song matching '{search_query}'", "artist": "Various Artists", "genre": "Pop", "mood": "happy"},
                        {"title": f"Another match for '{search_query}'", "artist": "Demo Artist", "genre": "Rock", "mood": "energetic"}
                    ]
                    
                    for result in search_results:
                        st.markdown(f"""
                        <div class="music-card">
                            <h4>🎵 {result['title']}</h4>
                            <p>👤 {result['artist']} | 🎭 {result['genre']} | 😊 {result['mood']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        with feature_tabs[1]:  # Music Discovery
            st.subheader("🎲 Music Discovery")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎲 Random Song Discovery"):
                    if songs_df is not None and not songs_df.empty:
                        random_song = songs_df.sample(1).iloc[0]
                        st.success("🎵 Here's a random discovery!")
                        st.markdown(f"""
                        <div class="music-card fade-in">
                            <h4>🎵 {random_song['title']}</h4>
                            <p>👤 {random_song['artist']} | 🎭 {random_song['genre']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🌟 Mood-based Discovery"):
                    current_mood = st.session_state.user_preferences["current_mood"]
                    discovery_recs = recommend_for_emotion(current_mood, top_k=3)
                    if not discovery_recs.empty:
                        st.success(f"🌟 Discoveries for {current_mood} mood!")
                        for _, song in discovery_recs.iterrows():
                            st.markdown(f"🎵 **{song['title']}** by {song['artist']}")
        
        with feature_tabs[2]:  # Trends
            st.subheader("📈 Music Trends & Analytics")
            
            if st.session_state.user_preferences["listening_history"]:
                history_df = pd.DataFrame(st.session_state.user_preferences["listening_history"])
                
                # Genre distribution
                genre_counts = history_df['genre'].value_counts()
                st.subheader("🎭 Your Genre Preferences")
                
                for genre, count in genre_counts.items():
                    percentage = (count / len(history_df)) * 100
                    st.progress(percentage / 100)
                    st.write(f"**{genre.title()}**: {count} plays ({percentage:.1f}%)")
                
                # Mood trends
                st.subheader("😊 Mood Trends")
                mood_counts = history_df['mood'].value_counts()
                for mood, count in mood_counts.head(3).items():
                    st.write(f"**{mood.title()}**: {count} times")
            else:
                st.info("📊 Listen to more music to see your trends!")
        
        with feature_tabs[3]:  # Fun Features
            st.subheader("🎪 Fun Music Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎵 Song of the Day")
                if st.button("🌅 Get Today's Song"):
                    if songs_df is not None and not songs_df.empty:
                        # Use date as seed for consistent daily song
                        today = datetime.now().date()
                        np.random.seed(hash(str(today)) % 2**32)
                        daily_song = songs_df.sample(1).iloc[0]
                        
                        st.success("🌅 Your song for today!")
                        st.markdown(f"""
                        <div class="music-card fade-in" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
                            <h4>🎵 {daily_song['title']}</h4>
                            <p>👤 {daily_song['artist']} | 🎭 {daily_song['genre']}</p>
                            <p><em>Perfect for {today.strftime('%A, %B %d')}</em></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("🎯 Music Mood Match")
                if st.button("🎯 Find My Perfect Match"):
                    current_mood = st.session_state.user_preferences["current_mood"]
                    perfect_match = recommend_for_emotion(current_mood, top_k=1)
                    
                    if not perfect_match.empty:
                        match = perfect_match.iloc[0]
                        st.success("🎯 Perfect match found!")
                        st.markdown(f"""
                        <div class="music-card fade-in" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                            <h4>💯 {match['title']}</h4>
                            <p>👤 {match['artist']}</p>
                            <p>🎯 Match Score: {match.get('score', 0):.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()