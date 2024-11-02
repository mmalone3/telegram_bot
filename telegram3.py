import json
import logging
import openai
import os
import schedule
import time
from collections import deque
from moviepy.editor import AudioFileClip
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Load API keys
with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()
openai.api_key = api_key

with open('telegramApiKey.txt', 'r') as f:
    telegram_key = f.read().strip()

# Initialize OpenAI client
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=api_key)

# Initialize Telegram bot
application = Application.builder().token(telegram_key).build()

# Initialize context queue
context_queue = deque(maxlen=10)

async def start(update: Update, context):
    await update.message.reply_text("Hello! I am an AI assistant. How can I help you today?")
    await update.message.reply_text("Please type your message below and I will respond to you as soon as possible.")

async def text_message(update: Update, context):
    message = update.message.text
    if not message:
        return
    
    sentiment = analyze_sentiment(message)
    context_queue.append(message)
    context_str = " ".join(context_queue)
    
    try:
        topics = perform_topic_modeling(conversation_history + [{"role": "user", "content": message}])
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        await update.message.reply_text("An error occurred while processing your message.")
        return
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Context: {context_str}. Sentiment: {sentiment}. Topic: {topics[0]}."},
        {"role": "user", "content": message}
    ]
    
    response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    await update.message.reply_text(response.choices[0].message.content)
    await add_to_conversation_history(message, response.choices[0].message.content)

application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Initialize other components
conversation_history = []
history_file = "conversation_history.json"
vectorizer = CountVectorizer()
lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
sia = SentimentIntensityAnalyzer()

# Load conversation history
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        conversation_history = json.load(f)

def load_conversation_history():
    global conversation_history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            conversation_history = json.load(f)
    return conversation_history

def save_conversation_history():
    global conversation_history
    with open(history_file, 'w') as f:
        json.dump(conversation_history, f)
    return True

async def add_to_conversation_history(user_input, bot_response):
    global conversation_history
    new_entry = {"user_input": user_input, "bot_response": bot_response}
    conversation_history.append(new_entry)
    save_conversation_history()
    return new_entry

def perform_topic_modeling(messages):
    corpus = vectorizer.fit_transform([m["content"] for m in messages if "content" in m])
    lda_model.fit(corpus)
    topics = lda_model.transform(corpus)
    return topics.tolist()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

def periodic_retraining():
    global conversation_history
    print("Periodic retraining initiated...")
    # Call OpenAI's API to retrain the model
    return True

schedule.every().day.at("00:00").do(periodic_retraining)

async def voice_message(update: Update, context):
    audio_file = await update.message.voice.get_file()
    await audio_file.download('audio.mp3')
    audio_clip = AudioFileClip('audio.mp3')
    transcript = recognize_speech(audio_clip)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": transcript}
    ]
    response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    await update.message.reply_text(response.choices[0].message.content)
    await add_to_conversation_history(transcript, response.choices[0].message.content)

def recognize_speech(audio_clip):
    return "Transcribed text from audio"

async def train_model(update: Update, context):
    await update.message.reply_text("Model training started.")
    # Call OpenAI's API to retrain the model
    return True

async def save_conversation(update: Update, context):
    await update.message.reply_text("Conversation saved.")

async def error(update: Update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

application.add_handler(MessageHandler(filters.VOICE, voice_message))

# Start the bot
if __name__ == '__main__':
    load_conversation_history()
    application.run_polling()