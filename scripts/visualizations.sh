python -m src.clf.analyse --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style_topic_pk_constraint \
--test_batch_size 1  \
--feat_extract \
--num_labels 2 \
--max_seq_length 500 \
--gpus 1 \
--task history_style \
--seed 36 \
--col_name tweet \
--data constraint \
--test_path "data/test.tsv" \
--output_fname "data/36_results_pk_style_topic_constraint_cos_sims.csv" \
--history_test_path "data/topic_semantic_test_results.tsv" \

#python -m src.clf.analyse \
#--visualize \
#--feat_path "data/recovery/article/36_results_topic_style_recovery_article_feats.csv" \
#--test_path "data/recovery/article/test.tsv" \
#--output_fname "data/recovery/article/36_results_topic_style_recovery_article_tsne.png"