import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from telegram3 import (
    text_message, analyze_sentiment, perform_topic_modeling, add_to_conversation_history
)

class TestTextMessage(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.update = MagicMock()
        self.context = MagicMock()
        self.update.message.text = "Test message"
        self.update.message.reply_text = AsyncMock()
        self.sentiment = "neutral"
        self.topics = [0.1]

    @patch('telegram3.add_to_conversation_history', new_callable=AsyncMock)
    @patch('telegram3.perform_topic_modeling', return_value=[[0.1]])
    @patch('telegram3.analyze_sentiment', return_value="neutral")
    @patch('openai.ChatCompletion.acreate', new_callable=AsyncMock)
    async def test_text_message(self, mock_create, mock_analyze_sentiment, mock_perform_topic_modeling, mock_add_to_conversation_history):
        mock_create.return_value = AsyncMock(choices=[{'message': {'content': 'Test response'}}])
        
        await text_message(self.update, self.context)
        
        self.update.message.reply_text.assert_called_with('Test response')
        mock_analyze_sentiment.assert_called_with("Test message")
        mock_perform_topic_modeling.assert_called()
        mock_add_to_conversation_history.assert_called()

if __name__ == '__main__':
    unittest.main()