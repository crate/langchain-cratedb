"""Test chat model integration."""

from typing import Type

from langchain_cratedb.chat_models import ChatCrateDB
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatCrateDBUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatCrateDB]:
        return ChatCrateDB

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
