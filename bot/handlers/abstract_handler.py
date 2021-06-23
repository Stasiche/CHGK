import logging
from abc import ABC
from telegram import Update
from telegram.ext import CallbackContext, Handler, CommandHandler


class AbstractHandler(ABC):
    def __init__(self):
        self.__logger = logging.getLogger(__file__)

    @property
    def command_name(self) -> str:
        raise NotImplementedError()

    def _callback(self, update: Update, callback_context: CallbackContext):
        raise NotImplementedError()

    def _callback_wrapper(self, update: Update, callback_context: CallbackContext):
        if update.effective_chat is None:
            self.__logger.error(f"Can't find source chat for {self.command_name} query")
            return
        self.__logger.info(
            f"Get {self.command_name} query from {update.effective_chat.id} ({update.effective_chat.username})"
        )
        self._callback(update, callback_context)

    def create(self) -> Handler:
        return CommandHandler(self.command_name, self._callback_wrapper)