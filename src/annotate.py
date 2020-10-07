from dataclasses import dataclass, field

@dataclass
class Post:
    _id: int
    label: str
    text: str
    reference_sources:list = field(default_factory=list)
    similar_claims:list = field(default_factory=list)

@dataclass
class ReferenceSource:
    domain: str
    label: str


def annotate(dataframe_obj):
    annotated_posts = []

    for _, row in dataframe_obj.iterrows():
        annotated_posts.append(Post(
            _id=row['id'],
            label = row['label'],
            text = row['tweet']
        ))

    return annotated_posts
    