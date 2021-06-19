from telegram.ext import CallbackContext
from telegram import Update
from handlers.abstract_handler import AbstractHandler
from model import Model


class ResetHandler(AbstractHandler):
    __message = "Хорошо, давай начнём сначала..."
    def __init__(self, model: Model):
        super().__init__()
        self.__model = model

    @property
    def command_name(self):
        return "reset"

    def _callback(self, update: Update, callback_context: CallbackContext):
        if update.effective_chat is None:
            self.__logger.error(f"Can't find source chat for {self.command_name} query")
            return
        self.__model.reset_user(update.effective_chat.id)
        callback_context.bot.send_message(chat_id=update.effective_chat.id, text=self.__message)