from telegram.ext import CallbackContext
from telegram import Update
from handlers.abstract_handler import AbstractHandler


class StartHandler(AbstractHandler):
    __start_message = (
        "Привет! Я бот, который поможет тебе насладится игрой настоящих интеллектуалов, элит и теневых правителей этого мира — ЧГК. Можешь попросить меня создать игру из готовых вопросов или найти вопрос на тему. Также наши видюхи всегда к твоим услугам — делают бррр и генерируют тебе вопрос, попробуй! Если забыл что к чему, бей в /help.\n\n"
        "И помни: согласно известному утверждению, еж --- это не только еж, но и ежиха. А что это за еж?"
    )

    @property
    def command_name(self):
        return "start"

    def _callback(self, update: Update, callback_context: CallbackContext):
        if update.effective_chat is None:
            self.__logger.error(f"Can't find source chat for {self.command_name} query")
            return
        callback_context.bot.send_message(chat_id=update.effective_chat.id, text=self.__start_message)
        