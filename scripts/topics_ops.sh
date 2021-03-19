#python -m src.pk.topics_explorer \
#--data constraint \
#--topic_model_path data/fakenews_topics.model \
#--data_path data \
#--top_k 10 \
#--target fake \

#python -m src.pk.topics_explorer \
#--data constraint \
#--topic_model_path data/fakenews_topics.model \
#--data_path data \
#--top_k 10 \
#--target fake \

python -m src.pk.topics_explorer \
--data coaid \
--topic_model_path data/fakenews_topics.model \
--data_path data/coaid/article/0 \
--top_k 10 \
--target fake \

#python -m src.pk.topics_explorer \
#--data coaid \
#--topic_model_path data/fakenews_topics.model \
#--data_path data/coaid/claims/0 \
#--top_k 10 \
#--target true \

#python -m src.pk.topics_explorer \
#--data recovery \
#--topic_model_path data/fakenews_topics.model \
#--data_path data/recovery/article \
#--top_k 10 \
#--target fake \

#python -m src.pk.topics_explorer \
#--data covid_tweets \
#--topic_model_path data/fakenews_topics.model \
#--data_path data/covid_tweets \
#--top_k 10 \
#--target fake \