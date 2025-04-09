"""
Unit tests for the agent module, including tests for the rewrite, generate, 
and run_rag_agent functions. These tests use mocking to simulate external 
dependencies and validate the behavior of the agent functions.
"""

import unittest
from unittest.mock import patch, MagicMock
from agent import run_rag_agent, rewrite, generate

class TestAgent(unittest.TestCase):
    """
    Test suite for the agent module. This class contains unit tests for the 
    rewrite, generate, and run_rag_agent functions, ensuring their correctness 
    and proper interaction with mocked dependencies.
    """
    @patch("agent.ChatOpenAI")
    def test_rewrite(self, mock_chat_openai):
        """
        Test the rewrite function to ensure it correctly processes the input 
        state and returns the expected rewritten message using the mocked 
        ChatOpenAI model.
        """
        # Mock the ChatOpenAI response
        mock_model = mock_chat_openai.return_value
        mock_model.invoke.return_value = MagicMock(content="Rewritten question")

        state = {"messages": [MagicMock(content="Original question")]}
        result = rewrite(state)

        self.assertEqual(result["messages"][0].content, "Rewritten question")
        mock_model.invoke.assert_called_once()

    @patch("agent.ChatOpenAI")
    @patch("agent.hub.pull")
    def test_generate(self, mock_hub_pull, mock_chat_openai):
        """
        Test the generate function to ensure it correctly processes the input 
        state, interacts with the mocked hub and ChatOpenAI model, and returns 
        the expected generated message.
        """
        # Mock the hub prompt and ChatOpenAI response
        mock_prompt = MagicMock()
        mock_hub_pull.return_value = mock_prompt
        mock_model = mock_chat_openai.return_value
        mock_response = MagicMock()
        mock_response.content = "Generated answer"
        mock_model.invoke.return_value = mock_response

        # Ensure the chain of calls in generate is properly mocked
        mock_hub_pull.return_value.__or__.return_value.__or__.return_value = mock_model

        state = {
            "messages": [
                MagicMock(content="User question"),
                MagicMock(content="Retrieved documents"),
            ]
        }
        result = generate(state)

        self.assertEqual(result["messages"][0].content, "Generated answer")
        mock_hub_pull.assert_called_once_with("rlm/rag-prompt")
        mock_model.invoke.assert_called_once()

    @patch("agent.graph.stream")
    def test_run_rag_agent(self, mock_stream):
        """
        Test the run_rag_agent function to ensure it processes the input 
        question, interacts with the mocked graph stream, and returns the 
        expected final answer.
        """
        # Mock the graph stream output
        mock_stream.return_value = iter([
            {"node1": {"messages": [MagicMock(content="Intermediate message")]}},
            {"node2": {"messages": [MagicMock(content="Final answer")]}},
        ])

        question = "What is the Sber 2023 report about?"
        result = run_rag_agent(question)

        self.assertEqual(result, "Final answer")
        mock_stream.assert_called_once()

if __name__ == "__main__":
    unittest.main()
