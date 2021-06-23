from dfa.base_state import BaseState, MoveResponse
from dfa.handler import DfaUserHandler
from logging import getLogger
from typing import List
from parse.intent import Intent
from parse.parser import Parser


class Model:
    def __init__(self):
        self.__logger = getLogger(__file__)
        self.__dfa_user_handler = DfaUserHandler()
        self.__parser = Parser()

    def handle_message(self, user_id: int, message: str) -> List[str]:
        intent = self.__parser.parse(message)
        current_state = self.__dfa_user_handler.get_user_dfa(user_id)
        response_messages: List[str] = []

        response = self.__make_move(user_id, current_state, intent, response_messages)
        while response.new_state.is_technical_state:
            response = self.__make_move(user_id, response.new_state, intent, response_messages)

        self.__dfa_user_handler.set_user_dfa(user_id, response.new_state)
        if response.new_state.introduce_message is not None:
            response_messages.append(response.new_state.introduce_message)
        return response_messages

    def __make_move(
        self, user_id: int, current_state: BaseState, intent: Intent, response_messages: List[str]
    ) -> MoveResponse:
        response = current_state.move(intent)
        self.__logger.info(
            f"Move {user_id} from {current_state.__class__.__name__} to {response.new_state.__class__.__name__}"
        )
        if response.message is not None:
            response_messages.append(response.message)
        return response

    def reset_user(self, user_id: int):
        self.__dfa_user_handler.reset_user(user_id)

    def get_state(self, user_id: int) -> str:
        current_state = self.__dfa_user_handler.get_user_dfa(user_id)
        return current_state.__class__.__name__
