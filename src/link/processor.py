from urlextract import URLExtract
from url_normalize import url_normalize
import requests
from cachetools import cached, TTLCache
from ..logger import logger
import tldextract


class URLExtractor:
    def __init__(self):
        self.extractor = URLExtract()

    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def normalize(self, url):
        if url.endswith('.'):
            url = url[: -1]
        try:
            session = requests.Session()
            resp = session.head(url, allow_redirects=True)
            unshorten_url = resp.url
            return url_normalize(unshorten_url)
        except requests.exceptions.RequestException as e:
            logger.error('Error {e} {url}'.format(e=e, url=url))
            return 'invalid_url'

    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def extract_url_metadata(self, url: str):
        res = tldextract.extract(url)
        return {'subdomain': res.subdomain,
                'domain': res.domain,
                'suffix': res.suffix,
                'alias': res.registered_domain}

    def transform(self, post: str):
        normalized_url = []
        urls = self.extractor.find_urls(post)
        for url in urls:
            normalized_url.append(self.normalize(url))
        return normalized_url
