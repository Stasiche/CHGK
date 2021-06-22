#!/usr/bin/env python3

import logging
import os
from bot_controller import BotController


TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


if __name__ == "__main__":
    token = os.environ.get(TELEGRAM_BOT_TOKEN)
    if token is None:
        print(f'can\'t find token for bot in env variable "{TELEGRAM_BOT_TOKEN}"')
    else:
        bot_controller = BotController(token)
        bot_controller.start()
