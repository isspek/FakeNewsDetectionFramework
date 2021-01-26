#docker pull docker.elastic.co/elasticsearch/elasticsearch:7.6.1
#docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.1
#python -m src.elasticsearch.search --tweets data/val.tsv --index_file_path data/semantic.json --option semantic
#python -m src.elasticsearch.search --tweets data/val.tsv --index_file_path data/semantic.json --option default
#python -m src.elasticsearch.search --mode test --index_file_path data/semantic.json --option semantic

#python -m src.elasticsearch.search --mode test --index_file_path data/semantic.json --option semantic
python -m src.elasticsearch.extract_bio_entities