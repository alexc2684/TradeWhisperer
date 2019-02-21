import re

class Preprocessor:

    def __init__(self):
        self.model = None

    def preprocess(self, texts):
        texts = [re.sub(r'([^\s\w]|_)+', '', text, flags=re.UNICODE) for text in texts]
        return texts
