from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, NewsSyntaxParser
from natasha.morph.tagger import NewsMorphTagger
from parse.intent import Intent, Command
from parse.question_parser import QuestionParser
from parse.game_parser import GameParser
from parse.search_parser import SearchParser
from parse.goodbye_parser import GoodbyeParser
from parse.greetings_parser import GreetingsParser

class Parser:
    def __init__(self):
        self.__command_parsers = {Command.GOODBYE: GoodbyeParser(),
                                  Command.GREETINGS: GreetingsParser(),
                                  Command.GAME: GameParser(),
                                  Command.QUESTION: QuestionParser(),
                                  Command.SEARCH: SearchParser(),
                                  }
        
        self.__segmenter = Segmenter()
        # TODO: check if there are existing another embeddings
        emb = NewsEmbedding()
        self.__morph_tagger = NewsMorphTagger(emb)
        self.__morph_vocab = MorphVocab()
        self.__ner_tagger = NewsNERTagger(emb)
        self.__syntax_parser = NewsSyntaxParser(emb)

    def __natasha_preprocessing(self, message: str) -> Doc:
        message = message.title()
        document = Doc(message)
        document.segment(self.__segmenter)
        document.tag_morph(self.__morph_tagger)
        for token in document.tokens:
            token.lemmatize(self.__morph_vocab)
        document.tag_ner(self.__ner_tagger)
        document.parse_syntax(self.__syntax_parser)
        for span in document.spans:
            span.normalize(self.__morph_vocab)
        return document

    def parse(self, message: str) -> Intent:
        document = self.__natasha_preprocessing(message)
        for command, command_parser in self.__command_parsers.items():
            parse_results = command_parser.process(document)
            if parse_results is not None:
                return Intent(command, message, parse_results)
        return Intent(Command.UNKNOWN, message)
