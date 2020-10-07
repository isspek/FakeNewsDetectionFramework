from urlextract import URLExtract

class URLExtractor:
    def __init__(self, realpart, imagpart):
        self.extractor = URLExtract()

    def transform(self, post:str):
        self.extractor.find_urls(post)