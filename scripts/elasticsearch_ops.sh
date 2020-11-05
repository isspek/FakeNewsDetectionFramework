# docker pull docker.elastic.co/elasticsearch/elasticsearch:7.6.1
# docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.1
python -m src.elasticsearch.baseline --vclaims data/processed/FakeHealth.tsv --tweets data/val.tsv --predict-file 'data/predictions.tsv'

# python -m src.index.create_index --index_file=data/index.json --index_name=fakenews
# python -m src.index.create_documents.py --data=data/processed/FakeHealth.tsv --index_name=fakenews