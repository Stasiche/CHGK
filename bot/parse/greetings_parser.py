from typing import Dict, Optional
from natasha import Doc
from parse.command_parser import CommandParser
from parse.keywords import keywords_greetings, keywords_game, keywords_generation, keywords_search

class GreetingsParser(CommandParser):
    __keywords = keywords_greetings
    __keywords_game = keywords_game
    __keywords_generation = keywords_generation
    __keywords_search = keywords_search

    def __extract_theme(self, message: Doc) -> Optional[str]:
        for token in message.tokens:
            if token.lemma == "тема":
                return message.text[token.stop:].strip()
        return None

    def process(self, message: Doc) -> Optional[Dict[str, str]]:
        out = {}
        is_keywords = self.__keywords.search(message.text.lower())
        if is_keywords:
            out["greetings"] = True
        
            if any([t.lemma in self.__keywords_game for t in message.tokens]):
                out["game"] = True
                return out
            
            if self.__keywords_generation.search(message.text.lower()):
                out["generation"] = True
                theme = self.__extract_theme(message)
                if theme is not None:
                    out["theme"] = theme
                return out
            
            if any([t.lemma in self.__keywords_search for t in message.tokens]):
                out["search"] = True
                theme = self.__extract_theme(message)
                if theme is not None:
                    out["theme"] = theme
                return out
                
            return out
            
        return None
