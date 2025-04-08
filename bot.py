import os
import logging
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from langgraph.errors import GraphRecursionError

# Import the RAG pipeline function
from agent import run_rag_agent

###############################################################################
# 1. LOGGING SETUP: everything logs to rag_debug.log
###############################################################################
LOG_FILENAME = os.path.join(os.getcwd(), "bot.log")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("bot_logger")
logger.info("Logger initialized. Writing debug info to '%s'", LOG_FILENAME)

##############################################################################
# 2. Load environment variables (including BOT_TOKEN)
##############################################################################
load_dotenv()

##############################################################################
# 3. Telegram Handlers
##############################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle the /start command.

    This function is triggered when a user sends the /start command to the bot.
    It sends a greeting message to the user and logs the interaction.

    Args:
        update (Update): The incoming update from Telegram.
        context (ContextTypes.DEFAULT_TYPE): The context for the callback.
    """
    user_id = update.effective_user.id
    logger.info("User /start invoked by user_id=%s", user_id)
    await update.message.reply_text("Hello! I am a RAG-based bot. Ask me anything.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle all non-command text messages sent by users.

    This function processes user messages, passes them to the RAG pipeline,
    and sends the generated response back to the user. It also handles specific
    errors like GraphRecursionError and logs any unexpected exceptions.

    Args:
        update (Update): The incoming update from Telegram.
        context (ContextTypes.DEFAULT_TYPE): The context for the callback.
    """
    user_id = update.effective_user.id
    user_text = update.message.text
    logger.info("Received message from user_id=%s: %s", user_id, user_text)

    try:
        # Call the RAG pipeline to get the final answer
        answer = run_rag_agent(user_text)
        logger.info("Returning answer to user_id=%s: %s", user_id, answer)
        await update.message.reply_text(answer or "No answer produced.")
    
    except GraphRecursionError as e:
        # Catch the recursion limit error
        logger.exception(
            "Graph recursion limit reached for user_id=%s. Error: %s", user_id, str(e)
        )
        await update.message.reply_text("Your question is too complex. Please refine or simplify it.")
    
    except Exception as e:
        logger.exception("Error processing user_id=%s message: %s", user_id, e)
        await update.message.reply_text("An error occurred. Please try again later.")

##############################################################################
# 4. Main Bot Entry
##############################################################################
def main():
    """
    Main entry point for the Telegram bot.

    This function initializes the bot, loads the BOT_TOKEN from environment
    variables, sets up command and message handlers, and starts polling for
    updates from Telegram.

    Raises:
        ValueError: If BOT_TOKEN is not found in the environment variables.
    """
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise ValueError("BOT_TOKEN not found in environment variables!")

    logger.info("Starting Telegram Bot with token %s", bot_token)

    # Build the Telegram application
    application = ApplicationBuilder().token(bot_token).build()

    # Register the /start command
    application.add_handler(CommandHandler("start", start_command))
    # Register a text message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is polling... Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()
