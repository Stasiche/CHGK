from telegram.ext import CallbackContext
from telegram import Update
from handlers.abstract_handler import AbstractHandler


class StartHandler(AbstractHandler):
    __start_message = (
        "Привет! Я бот, который может сгенерировать тебе вопрос из ЧГК! "
        "Просто опиши мне в свободной форме, какую тему ты хочешь."
    )

    @property
    def command_name(self):
        return "start"

    def _callback(self, update: Update, callback_context: CallbackContext):
        if update.effective_chat is None:
            self.__logger.error(f"Can't find source chat for {self.command_name} query")
            return
        callback_context.bot.send_message(chat_id=update.effective_chat.id, text=self.__start_message)
        