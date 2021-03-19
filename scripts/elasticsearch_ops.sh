#docker pull docker.elastic.co/elasticsearch/elasticsearch:7.6.1
#docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.1
#python -m src.pk.search \
#--mode train \
#--data_path data/recovery/article/train.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data recovery \
#--output_dir data/recovery/article \
#--col_name content
#
#python -m src.pk.search \
#--mode val \
#--data_path data/recovery/article/val.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data recovery \
#--output_dir data/recovery/article \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/recovery/article/test.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data recovery \
#--output_dir data/recovery/article \
#--col_name content

#python -m src.pk.search \
#--mode test \
#--data_path data/covid_tweets/test.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data covid \
#--output_dir data/covid_tweets \
#--col_name content

#python -m src.pk.search \
#--mode train \
#--data_path data/coaid/article/36/train.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data coaid \
#--output_dir data/coaid/article/36/ \
#--col_name content
##
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/article/36/val.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data coaid \
#--output_dir data/coaid/article/36/ \
#--col_name content
##
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/article/36/test.tsv \
#--index_file_path data/semantic_bio.json \
#--option semantic_bio \
#--data coaid \
#--output_dir data/coaid/article/36/ \
#--col_name content
#
#
python -m src.pk.search \
--mode train \
--data_path data/coaid/claims/42/train.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/42/ \
--col_name content
#
python -m src.pk.search \
--mode val \
--data_path data/coaid/claims/42/val.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/42/ \
--col_name content
#
python -m src.pk.search \
--mode test \
--data_path data/coaid/claims/42/test.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/42/ \
--col_name content



python -m src.pk.search \
--mode train \
--data_path data/coaid/claims/36/train.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/36/ \
--col_name content
#
python -m src.pk.search \
--mode val \
--data_path data/coaid/claims/36/val.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/36/ \
--col_name content
#
python -m src.pk.search \
--mode test \
--data_path data/coaid/claims/0/test.tsv \
--index_file_path data/semantic_bio.json \
--option semantic_bio \
--data coaid \
--output_dir data/coaid/claims/0/ \
--col_name content

#
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/article/0/test.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/0/ \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/article/36/test.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/36/ \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/article/42/test.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/42/ \
#--col_name content
#
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/article/0/val.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/0/ \
#--col_name content
#
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/article/36/val.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/36/ \
#--col_name content
#
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/article/42/val.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/coaid/article/42/ \
#--col_name content

#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/claims/0/val.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/0/ \
#--col_name content
##
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/claims/36/val.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/36/ \
#--col_name content
#
#python -m src.pk.search \
#--mode val \
#--data_path data/coaid/claims/42/val.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/42/ \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/claims/0/test.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/0/ \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/claims/36/test.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/36/ \
#--col_name content
#
#python -m src.pk.search \
#--mode test \
#--data_path data/coaid/claims/42/test.tsv \
#--index_file_path data/semantic.json \
#--option semantic \
#--data coaid \
#--output_dir data/coaid/claims/42/ \
#--col_name content

#python -m src.pk.search \
#--mode train \
#--data_path data/train.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--col_name tweet \
#--output_dir data/ \
#
#python -m src.pk.search \
#--mode val \
#--data_path data/val.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/ \
#--col_name tweet
#
#python -m src.pk.search \
#--mode test \
#--data_path data/test.tsv \
#--index_file_path data/topic_semantic.json \
#--option topic_semantic \
#--data coaid \
#--output_dir data/ \
#--col_name tweet