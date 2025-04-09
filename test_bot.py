"""
Unit tests for the Telegram bot module. These tests validate the behavior of 
the start_command and handle_message functions, including their interactions 
with mocked dependencies and error handling scenarios.
"""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from telegram.ext import ContextTypes

from langgraph.errors import GraphRecursionError
from bot import start_command, handle_message


class TestTelegramBot(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the Telegram bot module. This class tests the behavior of 
    the start_command and handle_message functions, including their interactions 
    with mocked dependencies and error handling scenarios.
    """
    @patch("bot.logger")
    async def test_start_command(self, mock_logger):
        """
        Test the start_command function to ensure it sends the correct welcome 
        message to the user and logs the invocation of the /start command.
        """
        # Mock Update and Context
        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Call the start_command function
        await start_command(mock_update, mock_context)

        # Assertions
        mock_update.message.reply_text.assert_called_once_with(
            "Hello! I am a RAG-based bot. Ask me anything."
        )
        mock_logger.info.assert_called_with("User /start invoked by user_id=%s", 12345)

    @patch("bot.run_rag_agent")
    @patch("bot.logger")
    async def test_handle_message_success(self, mock_logger, mock_run_rag_agent):
        """
        Test the handle_message function to ensure it processes a user's message 
        successfully, calls the RAG agent, and sends the correct response back 
        to the user while logging the interaction.
        """
        # Mock Update and Context
        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.text = "What is AI?"
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Mock the RAG pipeline response
        mock_run_rag_agent.return_value = "AI stands for Artificial Intelligence."

        # Call the handle_message function
        await handle_message(mock_update, mock_context)

        # Assertions
        mock_run_rag_agent.assert_called_once_with("What is AI?")
        mock_update.message.reply_text.assert_called_once_with(
            "AI stands for Artificial Intelligence."
        )
        mock_logger.info.assert_any_call(
            "Received message from user_id=%s: %s", 12345, "What is AI?"
        )
        mock_logger.info.assert_any_call(
            "Returning answer to user_id=%s: %s", 12345, "AI stands for Artificial Intelligence."
        )

    @patch("bot.logger")
    async def test_handle_message_graph_recursion_error(self, mock_logger):
        """
        Test the handle_message function to ensure it handles the 
        GraphRecursionError correctly by sending an appropriate error 
        message to the user and logging the exception.
        """
        # Mock Update and Context
        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.text = "Complex question"
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Mock the RAG pipeline to raise GraphRecursionError
        with patch("bot.run_rag_agent", side_effect=GraphRecursionError("Recursion limit reached")):
            await handle_message(mock_update, mock_context)

        # Assertions
        mock_update.message.reply_text.assert_called_once_with(
            "Your question is too complex. Please refine or simplify it."
        )
        mock_logger.exception.assert_called_once_with(
            "Graph recursion limit reached for user_id=%s. Error: %s",
            12345,
            "Recursion limit reached",
        )

    async def test_handle_message_general_exception(self):
        """
        Test the handle_message function to ensure it handles general exceptions 
        correctly by sending a generic error message to the user and logging the 
        exception.
        """
        # Mock Update and Context
        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.text = "Another question"
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Mock the RAG pipeline to raise a general exception
        with patch("bot.run_rag_agent", side_effect=Exception("Unexpected error")):
            await handle_message(mock_update, mock_context)

        # Assertions
        mock_update.message.reply_text.assert_called_once_with(
            "An unexpected error occurred. Please try again later."
        )
        # mock_logger.exception.assert_called_once_with(
        #     "Unexpected error processing user_id=%s message: %s", 12345, "Unexpected error"
        # )

if __name__ == "__main__":
    unittest.main()
