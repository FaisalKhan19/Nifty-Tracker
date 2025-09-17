from telegram.ext import Application, CommandHandler
import configparser
import json
import asyncio
from datetime import datetime, timedelta
from Nifty_Tracker import Tracker

config = configparser.ConfigParser()
config.read('.ini')
# Replace with your bot token
BOT_TOKEN = config['Bot']['Token']
tracker = Tracker(config_path='.ini')
# Mailing list (in memory; you can later persist to a file/db)
subscribers = set()

def save_subscribers():
    with open("subscribers.json", "w") as f:
        json.dump(list(subscribers), f)
    f.close()
    
def load_subscribers():
    global subscribers
    try:
        with open("subscribers.json", "r") as f:
            content = f.read().strip()
            if content:  # Check if file is not empty
                subscribers = set(json.loads(content))
            else:
                subscribers = set()
    except (FileNotFoundError, json.JSONDecodeError):
        subscribers = set()

async def start(update, context):
    chat_id = update.message.chat_id
    subscribers.add(chat_id)
    save_subscribers()  # Save after adding
    await update.message.reply_text("✅ You're now subscribed to result alerts!")
    await update.message.reply_text("Performing a test now!")
    try:
        today = tracker.format_date(datetime.today())
        prev = tracker.format_date(datetime.today() - timedelta(days=2))
        tracker.getAnnouncements(code='1234', prevDate=prev, toDate=today, test=True)
        message, _ = tracker.getProcessed()
        await update.message.reply_text(message)
        await update.message.reply_text("Test successful")
    except Exception as e:
        print(e)
        await update.message.reply_text("Test failed")

async def stop(update, context):
    chat_id = update.message.chat_id
    subscribers.discard(chat_id)
    save_subscribers()  # Save after removing
    await update.message.reply_text("❌ You've unsubscribed from alerts.")

async def list_subscribers(update, context):
    await update.message.reply_text(f"Current subscribers: {len(subscribers)}")

def main():
    load_subscribers()
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("list", list_subscribers))  # optional admin command

    application.run_polling()

if __name__ == '__main__':
    main()
