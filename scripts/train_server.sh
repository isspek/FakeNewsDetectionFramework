#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_links_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_train \
#--seed 0 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json"


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_history_links_style_bio \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 42 \
#--val_path "data/val.tsv" \
#--history_val_path "data/semantic_bio_val_results.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--test_path "data/test.tsv" \
#--history_test_path "data/semantic_bio_test_results.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--do_train \
#--train_path "data/train.tsv" \
#--history_train_path "data/semantic_bio_train_results.tsv" \
#--link_train_path "data/train_links_processed.json" \

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_history_links_style_bio \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 36 \
#--val_path "data/val.tsv" \
#--history_val_path "data/semantic_bio_val_results.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--test_path "data/test.tsv" \
#--history_test_path "data/semantic_bio_test_results.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--do_train \
#--train_path "data/train.tsv" \
#--history_train_path "data/semantic_bio_train_results.tsv" \
#--link_train_path "data/train_links_processed.json" \

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_history_links \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv"
##--test_path "data/test.tsv" \
##--link_test_path "data/test_links_processed.json" \
##--history_test_path "data/semantic_test_results.tsv" \

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_topic_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task topic_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--train_path "data/recovery/article/train.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--data recovery
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_topic_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task topic_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--col_name content \
#--do_train \
#--train_path "data/recovery/article/train.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--data recovery
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_topic_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task topic_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/recovery/article/train.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--data recovery

python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_pk_style_bio_recovery_article \
--train_batch_size 1 \
--eval_batch_size 1 \
--test_batch_size 1  \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 500 \
--gpus 1 \
--task history_style \
--max_grad_norm 1 \
--num_train_epochs 3 \
--do_predict \
--seed 0 \
--col_name content \
--do_train \
--history_train_path "data/recovery/article/semantic_bio_train_results.tsv" \
--train_path "data/recovery/article/train.tsv" \
--history_val_path "data/recovery/article/semantic_bio_val_results.tsv" \
--val_path "data/recovery/article/val.tsv" \
--test_path "data/recovery/article/test.tsv" \
--history_test_path "data/recovery/article/semantic_bio_results.tsv" \
--data recovery
#
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_pk_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--col_name content \
#--do_train \
#--history_train_path "data/recovery/article/topic_semantic_train_results.tsv" \
#--train_path "data/recovery/article/train.tsv" \
#--history_val_path "data/recovery/article/topic_semantic_val_results.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--history_test_path "data/recovery/article/topic_semantic_test_results.tsv" \
#--data recovery
#
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_pk_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--history_train_path "data/recovery/article/topic_semantic_train_results.tsv" \
#--train_path "data/recovery/article/train.tsv" \
#--history_val_path "data/recovery/article/topic_semantic_val_results.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--history_test_path "data/recovery/article/topic_semantic_test_results.tsv" \
#--data recovery

##
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_pk_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--history_train_path "data/recovery/article/recovery_semantic_train_results.tsv" \
#--train_path "data/recovery/article/train.tsv" \
#--history_val_path "data/recovery/article/recovery_semantic_val_results.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--history_test_path "data/recovery/article/recovery_semantic_test_results.tsv" \
#--data recovery
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_pk_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--history_train_path "data/recovery/article/recovery_semantic_train_results.tsv" \
#--train_path "data/recovery/article/train.tsv" \
#--history_val_path "data/recovery/article/recovery_semantic_val_results.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--history_test_path "data/recovery/article/recovery_semantic_test_results.tsv" \
#--data recovery

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--train_path "data/recovery/article/train.tsv" \
#--val_path "data/recovery/article/val.tsv" \
#--test_path "data/recovery/article/test.tsv" \
#--data recovery


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/claims/42/train.tsv" \
#--history_train_path "data/coaid/claims/42/recovery_semantic_train_results.tsv" \
#--val_path "data/coaid/claims/42/val.tsv" \
#--history_val_path "data/coaid/claims/42/recovery_semantic_val_results.tsv" \
#--test_path "data/coaid/claims/42/test.tsv" \
#--history_test_path "data/coaid/claims/42/recovery_semantic_test_results.tsv" \
#--data coaid
#

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_pk_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/42/train.tsv" \
#--history_train_path "data/coaid/article/42/coaid_semantic_train_results.tsv" \
#--val_path "data/coaid/article/42/val.tsv" \
#--history_val_path "data/coaid/article/42/coaid_semantic_val_results.tsv" \
#--test_path "data/coaid/article/42/test.tsv" \
#--history_test_path "data/coaid/article/42/coaid_semantic_test_results.tsv" \
#--data coaid


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_history_links \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_history_links \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv"
#

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/claims/0/train.tsv" \
#--val_path "data/coaid/claims/0/val.tsv" \
#--test_path "data/coaid/claims/0/test.tsv" \
#--data coaid

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name tweet \
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv" \
#--test_path "data/test.tsv" \
#--data constraint
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 42 \
#--col_name content \
#--test_path "data/covid_tweets/test.tsv" \
#--output_fname "covid_tweets_results.csv" \
#--data coaid
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv" \
#
##
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name tweet \
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv" \
#--test_path "data/test.tsv" \
#--data constraint

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_topic_pk_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--col_name content \
#--data coaid \
#--test_path "data/covid_tweets/test.tsv" \
#--history_test_path "data/covid_tweets/topic_semantic_test_results.tsv" \
#--output_fname "covid_tweets_results.csv"
##
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style_topic_pk_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 36 \
#--col_name content \
#--data coaid \
#--test_path "data/covid_tweets/test.tsv" \
#--history_test_path "data/covid_tweets/topic_semantic_test_results.tsv" \
#--output_fname "covid_tweets_results.csv"
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_topic_pk_constraint \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--data coaid \
#--test_path "data/covid_tweets/test.tsv" \
#--history_test_path "data/covid_tweets/topic_semantic_test_results.tsv" \
#--output_fname "covid_tweets_results.csv"


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/claims/0/train.tsv" \
#--val_path "data/coaid/claims/0/val.tsv" \
#--test_path "data/coaid/claims/0/test.tsv" \
#--data coaid
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/claims/36/train.tsv" \
#--val_path "data/coaid/claims/36/val.tsv" \
#--test_path "data/coaid/claims/36/test.tsv" \
#--data coaid
#
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_coaid_claims \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/claims/42/train.tsv" \
#--val_path "data/coaid/claims/42/val.tsv" \
#--test_path "data/coaid/claims/42/test.tsv" \
#--data coaid


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style_ent_topic_pk_coaid_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 0 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/0/train.tsv" \
#--history_train_path "data/coaid/article/0/semantic_bio_train_results.tsv" \
#--val_path "data/coaid/article/0/val.tsv" \
#--history_val_path "data/coaid/article/0/semantic_bio_val_results.tsv" \
#--test_path "data/coaid/article/0/test.tsv" \
#--history_test_path "data/coaid/article/0/semantic_bio_test_results.tsv" \
#--data coaid



#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style_ent_topic_pk_coaid_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/36/train.tsv" \
#--history_train_path "data/coaid/article/36/semantic_bio_train_results.tsv" \
#--val_path "data/coaid/article/36/val.tsv" \
#--history_val_path "data/coaid/article/36/semantic_bio_val_results.tsv" \
#--test_path "data/coaid/article/36/test.tsv" \
#--history_test_path "data/coaid/article/36/semantic_bio_test_results.tsv" \
#--data coaid

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_ent_topic_pk_coaid_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/42/train.tsv" \
#--history_train_path "data/coaid/article/42/semantic_bio_train_results.tsv" \
#--val_path "data/coaid/article/42/val.tsv" \
#--history_val_path "data/coaid/article/42/semantic_bio_val_results.tsv" \
#--test_path "data/coaid/article/42/test.tsv" \
#--history_test_path "data/coaid/article/42/semantic_bio_test_results.tsv" \
#--data coaid

##
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style_topic_pk_coaid_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 36 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/36/train.tsv" \
#--history_train_path "data/coaid/article/36/topic_semantic_train_results.tsv" \
#--val_path "data/coaid/article/36/val.tsv" \
#--history_val_path "data/coaid/article/36/topic_semantic_val_results.tsv" \
#--test_path "data/coaid/article/36/test.tsv" \
#--history_test_path "data/coaid/article/36/topic_semantic_test_results.tsv" \
#--data coaid
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_topic_pk_coaid_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/42/train.tsv" \
#--history_train_path "data/coaid/article/42/topic_semantic_train_results.tsv" \
#--val_path "data/coaid/article/42/val.tsv" \
#--history_val_path "data/coaid/article/42/topic_semantic_val_results.tsv" \
#--test_path "data/coaid/article/42/test.tsv" \
#--history_test_path "data/coaid/article/42/topic_semantic_test_results.tsv" \
#--data coaid


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style_pk_recovery_article \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--test_batch_size 1  \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 500 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--col_name content \
#--do_train \
#--train_path "data/coaid/article/42/train.tsv" \
#--history_train_path "data/coaid/article/42/coaid_semantic_train_results.tsv" \
#--val_path "data/coaid/article/42/val.tsv" \
#--history_val_path "data/coaid/article/42/coaid_semantic_val_results.tsv" \
#--test_path "data/coaid/article/42/test.tsv" \
#--history_test_path "data/coaid/article/42/coaid_semantic_test_results.tsv" \
#--data coaid
