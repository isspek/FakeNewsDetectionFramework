from urlextract import URLExtract

class URLExtractor:
    def __init__(self):
        self.extractor = URLExtract()

    def transform(self, post:str):
        return self.extractor.find_urls(post)