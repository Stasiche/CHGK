from typing import Dict, Optional
from natasha import Doc
from parse.command_parser import CommandParser
from parse.keywords import keywords_goodbye

class GoodbyeParser(CommandParser):
    __keywords = keywords_goodbye

    def process(self, message: Doc) -> Optional[Dict[str, str]]:
        is_keywords = self.__keywords.search(message.text.lower())
        if is_keywords and len(message.tokens) <= 3:  # до свидания! == 3 токена
            return {"bye": True}
        return None
