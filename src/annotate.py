from dataclasses import dataclass, field
from .link.processor import URLExtractor

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
        print(urls)


@dataclass
class ReferenceSource:
    domain: str
    label: str


def annotate(dataframe_obj):
    annotated_posts = []

    for _, row in dataframe_obj.iterrows():
        annotated_posts.append(Post(
            _id=row['id'],
            label=row['label'],
            text=row['tweet']
        ))

    return annotated_posts
