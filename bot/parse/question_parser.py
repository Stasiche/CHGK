from typing import Dict, Optional
from natasha import Doc
from parse.command_parser import CommandParser
from parse.keywords import keywords_generation


class QuestionParser(CommandParser):
    __keywords = keywords_generation

    def __extract_theme(self, message: Doc) -> Optional[str]:
        for token in message.tokens:
            if token.lemma == "тема":
                return message.text[token.stop:].strip()
        return None

    def process(self, message: Doc) -> Optional[Dict[str, str]]:
        # is_keywords = any([t.lemma in self.__keywords for t in message.tokens])
        is_keywords = self.__keywords.search(message.text.lower())
        if is_keywords:
            theme = self.__extract_theme(message)
            if theme is not None:
                return {"theme": theme}
            return {}
        return None
