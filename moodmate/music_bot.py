import chainlit as cl
import json
import random
import openai
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
import re
from collections import defaultdict, Counter
import pickle
import os

class AILearningMusicBot:
    def __init__(self):
        # Enhanced music database with learning capabilities
        self.music_db = {
            "artists": {
                # International Artists
                "taylor swift": {
                    "genre": ["pop", "country", "indie"],
                    "popular_songs": ["Anti-Hero", "Shake It Off", "Love Story", "Blank Space"],
                    "albums": ["Midnights", "1989", "Folklore", "Red"],
                    "trivia": "Has won 12 Grammy Awards and is known for re-recording her early albums.",
                    "ai_learned_facts": []
                },
                "the beatles": {
                    "genre": ["rock", "pop"],
                    "popular_songs": ["Hey Jude", "Let It Be", "Yesterday", "Come Together"],
                    "albums": ["Abbey Road", "Sgt. Pepper's", "Revolver", "White Album"],
                    "trivia": "Formed in Liverpool in 1960 and are considered the most influential band of all time.",
                    "ai_learned_facts": []
                },
                "drake": {
                    "genre": ["hip-hop", "rap", "r&b"],
                    "popular_songs": ["God's Plan", "Hotline Bling", "In My Feelings", "One Dance"],
                    "albums": ["Scorpion", "Views", "Take Care", "Nothing Was the Same"],
                    "trivia": "Started as an actor on Degrassi before becoming one of the best-selling music artists.",
                    "ai_learned_facts": []
                },
                
                # Telugu Artists
                "s. p. balasubrahmanyam": {
                    "genre": ["telugu classical", "playback singing", "devotional"],
                    "popular_songs": ["Sankarabharanam", "Swathi Mutyam", "Sagara Sangamam", "Om Namaha Shivaya"],
                    "albums": ["Classical Collections", "Devotional Hits", "Film Songs"],
                    "trivia": "Legendary playback singer with over 40,000 songs in 16 languages. Holds Guinness World Record.",
                    "ai_learned_facts": []
                },
                "m. m. keeravani": {
                    "genre": ["telugu film music", "classical fusion", "instrumental"],
                    "popular_songs": ["Dheevara - Baahubali", "Naatu Naatu - RRR", "Keeravani Raag Songs"],
                    "albums": ["Baahubali Soundtrack", "RRR Soundtrack", "Magadheera Music"],
                    "trivia": "Oscar winner for 'Naatu Naatu' from RRR. Master of fusion music composition.",
                    "ai_learned_facts": []
                },
                "devi sri prasad": {
                    "genre": ["telugu pop", "film music", "energetic"],
                    "popular_songs": ["Buttabomma", "Seeti Maar", "Samajavaragamana", "Pushpa Theme"],
                    "albums": ["Pushpa Soundtrack", "Arya Music", "Gabbar Singh Songs"],
                    "trivia": "Known as 'Rockstar DSP' for his energetic and catchy Telugu film music.",
                    "ai_learned_facts": []
                }
            },
            "genres": {
                "rock": ["The Beatles", "Queen", "Led Zeppelin", "Pink Floyd", "AC/DC"],
                "pop": ["Taylor Swift", "Ariana Grande", "Ed Sheeran", "Dua Lipa", "The Weeknd"],
                "hip-hop": ["Drake", "Kendrick Lamar", "J. Cole", "Travis Scott", "Kanye West"],
                "jazz": ["Miles Davis", "John Coltrane", "Duke Ellington", "Billie Holiday"],
                "classical": ["Mozart", "Beethoven", "Bach", "Chopin", "Vivaldi"],
                "electronic": ["Daft Punk", "Calvin Harris", "Skrillex", "Deadmau5", "Avicii"],
                "telugu classical": ["S. P. Balasubrahmanyam", "K. J. Yesudas", "M. S. Subbulakshmi"],
                "telugu film music": ["Ilaiyaraaja", "M. M. Keeravani", "Devi Sri Prasad", "S. S. Thaman"],
                "tollywood": ["Devi Sri Prasad", "M. M. Keeravani", "S. S. Thaman", "Anirudh Ravichander"]
            },
            "moods": {
                "happy": ["Uptown Funk - Bruno Mars", "Buttabomma - Allu Arjun", "Can't Stop the Feeling - Justin Timberlake"],
                "sad": ["Someone Like You - Adele", "Nuvvostanante Nenoddantana - Title Song", "Hurt - Johnny Cash"],
                "energetic": ["Thunder - Imagine Dragons", "Naatu Naatu - RRR", "Pushpa Raj Theme"],
                "relaxed": ["Weightless - Marconi Union", "Swathi Mutyam - Classical", "Claire de Lune - Debussy"],
                "romantic": ["Perfect - Ed Sheeran", "Inkem Inkem - Geetha Govindam", "All of Me - John Legend"]
            }
        }
        
        # Learning system components
        self.user_interactions = defaultdict(list)  # Track user conversation history
        self.user_preferences = defaultdict(dict)   # Track user music preferences
        self.learned_songs = defaultdict(list)      # Songs learned from users
        self.learned_artists = defaultdict(dict)    # Artists learned from users
        self.mood_patterns = defaultdict(list)      # User mood-music patterns
        self.conversation_context = defaultdict(list)  # Maintain conversation context
        
        # AI response templates for more natural conversation
        self.response_templates = {
            "greeting": [
                "Hey there! 🎵 What kind of music adventure are we going on today?",
                "Welcome back! 🎶 I've been learning about music - what can I help you discover?",
                "Hi! 🎤 Ready to explore some amazing music together?"
            ],
            "recommendation_intro": [
                "Based on what you're feeling, here are some perfect tracks:",
                "I think you'll love these songs for your current mood:",
                "Here's what I'd recommend based on your vibe:"
            ],
            "learning_response": [
                "Thanks for teaching me about that! I'll remember this for future recommendations.",
                "That's fascinating! I'm adding this to my music knowledge.",
                "I love learning new things about music from you!"
            ]
        }
        
        # Load previous learning data
        self.load_learning_data()
    
    def save_learning_data(self):
        """Save learned data to persistent storage"""
        learning_data = {
            'user_preferences': dict(self.user_preferences),
            'learned_songs': dict(self.learned_songs),
            'learned_artists': dict(self.learned_artists),
            'mood_patterns': dict(self.mood_patterns)
        }
        
        try:
            with open('music_bot_learning.json', 'w') as f:
                json.dump(learning_data, f, indent=2)
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def load_learning_data(self):
        """Load previously learned data"""
        try:
            if os.path.exists('music_bot_learning.json'):
                with open('music_bot_learning.json', 'r') as f:
                    learning_data = json.load(f)
                    
                self.user_preferences = defaultdict(dict, learning_data.get('user_preferences', {}))
                self.learned_songs = defaultdict(list, learning_data.get('learned_songs', {}))
                self.learned_artists = defaultdict(dict, learning_data.get('learned_artists', {}))
                self.mood_patterns = defaultdict(list, learning_data.get('mood_patterns', {}))
        except Exception as e:
            print(f"Error loading learning data: {e}")
    
    def analyze_user_input(self, user_input: str, user_id: str) -> Dict:
        """Advanced analysis of user input using AI techniques"""
        analysis = {
            'intent': None,
            'entities': [],
            'mood': None,
            'genre': None,
            'artist': None,
            'song': None,
            'sentiment': 'neutral',
            'learning_opportunity': False
        }
        
        user_input_lower = user_input.lower()
        
        # Intent classification
        if any(word in user_input_lower for word in ['feeling', 'mood', 'vibe']):
            analysis['intent'] = 'mood_request'
        elif any(word in user_input_lower for word in ['recommend', 'suggest', 'play']):
            analysis['intent'] = 'recommendation_request'
        elif any(word in user_input_lower for word in ['tell me about', 'who is', 'info about']):
            analysis['intent'] = 'information_request'
        elif any(word in user_input_lower for word in ['love', 'like', 'favorite', 'best']):
            analysis['intent'] = 'preference_sharing'
            analysis['learning_opportunity'] = True
        elif any(word in user_input_lower for word in ['create', 'add', 'playlist']):
            analysis['intent'] = 'playlist_management'
        
        # Mood detection
        mood_indicators = {
            'happy': ['happy', 'joyful', 'excited', 'cheerful', 'upbeat', 'good', 'great', 'awesome'],
            'sad': ['sad', 'depressed', 'down', 'blue', 'upset', 'crying', 'heartbroken'],
            'energetic': ['energetic', 'pumped', 'hyped', 'active', 'workout', 'gym', 'dance'],
            'relaxed': ['relaxed', 'chill', 'calm', 'peaceful', 'zen', 'mellow', 'tired'],
            'romantic': ['romantic', 'love', 'loving', 'valentine', 'crush', 'date']
        }
        
        for mood, indicators in mood_indicators.items():
            if any(indicator in user_input_lower for indicator in indicators):
                analysis['mood'] = mood
                break
        
        # Genre detection
        for genre in self.music_db['genres'].keys():
            if genre in user_input_lower or genre.replace(' ', '') in user_input_lower:
                analysis['genre'] = genre
                break
        
        # Artist detection
        for artist in self.music_db['artists'].keys():
            if artist in user_input_lower:
                analysis['artist'] = artist
                break
        
        # Store interaction for learning
        self.user_interactions[user_id].append({
            'input': user_input,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return analysis
    
    def generate_ai_response(self, analysis: Dict, user_id: str) -> str:
        """Generate contextual AI responses based on analysis"""
        
        # Check for learning opportunities
        if analysis['learning_opportunity']:
            return self.handle_learning_opportunity(analysis, user_id)
        
        # Handle different intents
        if analysis['intent'] == 'mood_request' and analysis['mood']:
            return self.get_personalized_mood_recommendations(analysis['mood'], user_id)
        elif analysis['intent'] == 'recommendation_request':
            return self.get_smart_recommendations(analysis, user_id)
        elif analysis['intent'] == 'information_request' and analysis['artist']:
            return self.get_enhanced_artist_info(analysis['artist'], user_id)
        else:
            return self.get_conversational_response(analysis, user_id)
    
    def handle_learning_opportunity(self, analysis: Dict, user_id: str) -> str:
        """Learn from user preferences and feedback"""
        user_input = analysis.get('original_input', '')
        
        # Extract what the user likes
        if 'love' in user_input.lower() or 'favorite' in user_input.lower():
            # Try to extract song/artist names
            potential_songs = re.findall(r'"([^"]*)"', user_input)  # Songs in quotes
            potential_songs.extend(re.findall(r"'([^']*)'", user_input))  # Songs in single quotes
            
            for song in potential_songs:
                if song not in self.learned_songs[user_id]:
                    self.learned_songs[user_id].append(song)
            
            # Update user preferences
            if analysis['mood']:
                if analysis['mood'] not in self.user_preferences[user_id]:
                    self.user_preferences[user_id][analysis['mood']] = []
                self.user_preferences[user_id][analysis['mood']].extend(potential_songs)
        
        self.save_learning_data()
        return random.choice(self.response_templates['learning_response']) + f"\n\nI've noted your preferences for future recommendations! 🎵"
    
    def get_personalized_mood_recommendations(self, mood: str, user_id: str) -> str:
        """Get mood recommendations personalized to user"""
        base_recommendations = self.music_db['moods'].get(mood, [])
        
        # Add user's previously liked songs for this mood
        user_mood_prefs = self.user_preferences[user_id].get(mood, [])
        
        # Combine and personalize
        all_recommendations = base_recommendations + user_mood_prefs
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for song in all_recommendations:
            if song not in seen:
                seen.add(song)
                unique_recommendations.append(song)
        
        response = f"🎵 **Perfect songs for when you're feeling {mood}:**\n\n"
        
        # Show personalized recommendations first
        if user_mood_prefs:
            response += "**🌟 Based on your preferences:**\n"
            for i, song in enumerate(user_mood_prefs[:3], 1):
                response += f"{i}. {song}\n"
            response += "\n**🎶 You might also like:**\n"
            start_num = len(user_mood_prefs[:3]) + 1
        else:
            start_num = 1
            
        for i, song in enumerate(base_recommendations[:5], start_num):
            response += f"{i}. {song}\n"
        
        response += f"\n💡 *Tell me if you love any of these - I'll remember for next time!*"
        
        # Store the mood pattern
        self.mood_patterns[user_id].append({
            'mood': mood,
            'timestamp': datetime.now().isoformat(),
            'recommendations': unique_recommendations[:8]
        })
        
        return response
    
    def get_smart_recommendations(self, analysis: Dict, user_id: str) -> str:
        """Get smart recommendations based on user history and context"""
        recommendations = []
        
        # Check user's historical preferences
        user_prefs = self.user_preferences[user_id]
        
        if analysis['genre']:
            # Genre-based recommendations
            genre_artists = self.music_db['genres'].get(analysis['genre'], [])
            recommendations.extend(genre_artists[:5])
            
        elif user_prefs:
            # Recommend based on user's past preferences
            most_common_mood = max(user_prefs.keys(), key=lambda k: len(user_prefs[k]))
            recommendations.extend(self.music_db['moods'].get(most_common_mood, [])[:5])
        
        else:
            # Default recommendations - mix of popular songs
            all_moods = ['happy', 'energetic', 'romantic']
            for mood in all_moods:
                recommendations.extend(self.music_db['moods'].get(mood, [])[:2])
        
        response = "🎶 **Here are some recommendations for you:**\n\n"
        for i, rec in enumerate(recommendations[:8], 1):
            response += f"{i}. {rec}\n"
        
        response += "\n💡 *These are based on your music taste - let me know what you think!*"
        return response
    
    def get_enhanced_artist_info(self, artist: str, user_id: str) -> str:
        """Get enhanced artist information including learned facts"""
        artist_info = self.music_db['artists'].get(artist, {})
        
        if not artist_info:
            return f"🤔 I don't have information about {artist.title()} yet, but I'd love to learn! Tell me something interesting about them."
        
        response = f"🎤 **{artist.title()}**\n\n"
        response += f"**Genres:** {', '.join(artist_info['genre'])}\n\n"
        response += f"**Popular Songs:**\n"
        for song in artist_info['popular_songs']:
            response += f"• {song}\n"
        response += f"\n**Albums:** {', '.join(artist_info['albums'])}\n\n"
        response += f"**Fun Fact:** {artist_info['trivia']}\n"
        
        # Add learned facts if any
        learned_facts = artist_info.get('ai_learned_facts', [])
        if learned_facts:
            response += f"\n**What I've learned from users:**\n"
            for fact in learned_facts[-3:]:  # Show last 3 learned facts
                response += f"• {fact}\n"
        
        response += f"\n💡 *Know something cool about {artist.title()}? Share it with me!*"
        return response
    
    def get_conversational_response(self, analysis: Dict, user_id: str) -> str:
        """Generate conversational responses for general queries"""
        
        # Context-aware responses
        recent_interactions = self.user_interactions[user_id][-3:] if self.user_interactions[user_id] else []
        
        if recent_interactions:
            last_interaction = recent_interactions[-1]
            if last_interaction['analysis'].get('mood'):
                return f"I remember you were feeling {last_interaction['analysis']['mood']} earlier. How are you feeling now? 🎵"
        
        # Default helpful response
        responses = [
            "🎵 I'm here to help you discover amazing music! What's your mood like today?",
            "🎶 Tell me about your musical preferences - I love learning about new artists and songs!",
            "🎤 Want to explore some music? I can recommend based on your mood, genre preferences, or favorite artists!",
            "🎼 I'm getting smarter about music every day thanks to conversations like ours. What would you like to explore?"
        ]
        
        return random.choice(responses)
    
    def add_learning_from_user(self, user_input: str, user_id: str):
        """Add new information shared by users to the knowledge base"""
        user_input_lower = user_input.lower()
        
        # Look for patterns like "X is a great Y artist" or "Did you know X released Y?"
        learning_patterns = [
            r"(.+) is a great (.+) artist",
            r"did you know (.+) released (.+)",
            r"(.+) makes amazing (.+) music",
            r"(.+) is known for (.+)"
        ]
        
        for pattern in learning_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                # Extract the information and add to learned facts
                fact = user_input.strip()
                
                # Find if it's about an existing artist
                for artist in self.music_db['artists'].keys():
                    if artist in user_input_lower:
                        if 'ai_learned_facts' not in self.music_db['artists'][artist]:
                            self.music_db['artists'][artist]['ai_learned_facts'] = []
                        
                        if fact not in self.music_db['artists'][artist]['ai_learned_facts']:
                            self.music_db['artists'][artist]['ai_learned_facts'].append(fact)
                            self.save_learning_data()
                            return True
                break
        
        return False
    
    async def process_message(self, message: str, user_id: str) -> str:
        """Main message processing with AI enhancement"""
        
        # Add original input to analysis for learning
        analysis = self.analyze_user_input(message, user_id)
        analysis['original_input'] = message
        
        # Try to learn from the input
        learned_something = self.add_learning_from_user(message, user_id)
        
        # Generate AI response
        response = self.generate_ai_response(analysis, user_id)
        
        # Add learning acknowledgment if something was learned
        if learned_something:
            response += f"\n\n✨ **Thanks for teaching me something new!** I'll remember this for future conversations."
        
        # Update conversation context
        self.conversation_context[user_id].append({
            'user_message': message,
            'bot_response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges for context
        if len(self.conversation_context[user_id]) > 10:
            self.conversation_context[user_id] = self.conversation_context[user_id][-10:]
        
        return response

# Initialize the AI music bot
ai_music_bot = AILearningMusicBot()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="""# 🎵 AI Music Bot - Now with Learning! 🧠🎵

I'm your intelligent music assistant that **learns from our conversations!**

## ✨ **What Makes Me Special:**
• 🧠 **I Learn From You** - Share your favorite songs, artists, or music facts!
• 🎯 **Personalized Recommendations** - I remember your preferences
• 🌍 **Global + Telugu Music** - From SPB to Taylor Swift, DSP to Drake
• 💬 **Natural Conversations** - Chat with me like a music-loving friend
• 📈 **Getting Smarter** - Every conversation makes my recommendations better

## 🎶 **How I Learn:**
• Tell me about songs you love: *"I love 'Shape of You' by Ed Sheeran"*
• Share music facts: *"Did you know AR Rahman won 2 Oscars?"*
• Express preferences: *"I'm really into Telugu classical music"*
• Give feedback: *"That recommendation was perfect!"*

## 💬 **Try Natural Conversations:**
• *"I'm feeling nostalgic, suggest some old Telugu songs"*
• *"My favorite artist is DSP, recommend similar composers"*
• *"Tell me something cool about SPB"*
• *"I love energetic workout music"*
• *"Create a playlist with my favorite romantic songs"*

## 🎯 **I Remember:**
• Your favorite moods and genres
• Songs and artists you mention
• Music facts you share
• Your playlist preferences

**Start by telling me about your music taste - I'm excited to learn about you!** 🎶✨
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()
    user_id = f"user_{hash(str(message.author))}" if message.author else "default_user"
    
    try:
        # Process message with AI enhancement
        response = await ai_music_bot.process_message(user_input, user_id)
        await cl.Message(content=response).send()
        
    except Exception as e:
        error_response = f"🤖 Oops! I encountered an error: {str(e)}\n\nBut hey, I'm always learning! Please try asking me something else about music. 🎵"
        await cl.Message(content=error_response).send()

if __name__ == "__main__":
    cl.run()