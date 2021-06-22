from telegram.ext import CallbackContext
from telegram import Update
from handlers.abstract_handler import AbstractHandler


class HelpHandler(AbstractHandler):
    __help_message = (
        "Сейчас у меня можно попросить:\n1) Сгенерировать вопрос. \n2) Найти вопрос в базе данных. \n3) Собрать игру из 12 вопросов\n\nЕсли я заблудился, то всегда можно сделать /reset"
    )

    @property
    def command_name(self):
        return "help"

    def _callback(self, update: Update, callback_context: CallbackContext):
        if update.effective_chat is None:
            self.__logger.error(f"Can't find source chat for {self.command_name} query")
            return
        callback_context.bot.send_message(chat_id=update.effective_chat.id, text=self.__help_message)
        