from typing import Dict, Optional
from natasha import Doc
from parse.command_parser import CommandParser
from parse.keywords import keywords_game


class GameParser(CommandParser):
    __keywords = keywords_game

    def process(self, message: Doc) -> Optional[Dict[str, str]]:
        is_keywords = any([t.lemma in self.__keywords for t in message.tokens])
        if is_keywords:
            return {"game": True}
        return None
