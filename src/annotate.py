from dataclasses import dataclass, field
from .link.processor import URLExtractor
from .logger import logger

url_extractor = URLExtractor()


@dataclass
class Post:
    _id: int
    label: str
    text: str
    linked_urls: list = field(default_factory=list)
    similar_claims: list = field(default_factory=list)

    def __post_init__(self):
        urls = url_extractor.transform(self.text)
        for url in urls:
            url_metadata = url_extractor.extract_url_metadata(url)
            logger.info('extracting referred sources...')
            self.linked_urls.append(ReferenceSource(
                subdomain=url_metadata['subdomain'],
                domain=url_metadata['domain'],
                suffix=url_metadata['suffix'],
                alias=url_metadata['alias']
            ))


@dataclass
class ReferenceSource:
    subdomain: str
    domain: str
    suffix: str
    alias: str
    label: str = 'notclassified'


def annotate(dataframe_obj):
    annotated_posts = []

    for _, row in dataframe_obj.iterrows():
        annotated_posts.append(Post(
            _id=row['id'],
            label=row['label'],
            text=row['tweet']
        ))

    return annotated_posts
