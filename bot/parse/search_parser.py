from typing import Dict, Optional
from natasha import Doc
from parse.command_parser import CommandParser
from parse.keywords import keywords_search


class SearchParser(CommandParser):
    __keywords = keywords_search

    def __extract_theme(self, message: Doc) -> Optional[str]:
        for token in message.tokens:
            if token.lemma == "тема":
                return message.text[token.stop:].strip()
        return None

    def process(self, message: Doc) -> Optional[Dict[str, str]]:
        is_keywords = any([t.lemma in self.__keywords for t in message.tokens])
        if is_keywords:
            theme = self.__extract_theme(message)
            if theme is not None:
                return {"theme": theme}
            return {}
        return None
