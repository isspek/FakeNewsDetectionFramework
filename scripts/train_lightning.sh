# python -m src.style.lightning_classifier --pretrained bert-base-uncased --auto_select_gpus true --batch_size 8 --learning_rate 2e-5 --gradient_clip_val 1.0 --nela_train data/NELA/train_1.tsv --nela_test data/NELA/test_1.tsv --num_labels 3 --max_len 512 \
# --gpus 1 --max_epochs 1 --task constraint



# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 2 \
# --max_seq_length 128 \
# --gpus 1 \
# --task constraint \
# --max_grad_norm 1 \
# --num_train_epochs 1 \
# --train_path "data/train.tsv" \
# --val_path "data/val.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \


# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_nela_1 \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 3 \
# --max_seq_length 510 \
# --gpus 1 \
# --task nela \
# --max_grad_norm 1 \
# --num_train_epochs 3 \
# --train_path "data/NELA/train_1.tsv" \
# --val_path "data/NELA/test_1.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \
#
# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_nela_2 \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 3 \
# --max_seq_length 510 \
# --gpus 1 \
# --task nela \
# --max_grad_norm 1 \
# --num_train_epochs 3 \
# --train_path "data/NELA/train_2.tsv" \
# --val_path "data/NELA/test_2.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \
#
# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_nela_3 \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 3 \
# --max_seq_length 510 \
# --gpus 1 \
# --task nela \
# --max_grad_norm 1 \
# --num_train_epochs 3 \
# --train_path "data/NELA/train_3.tsv" \
# --val_path "data/NELA/test_3.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \
#
#
# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_nela_4 \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 3 \
# --max_seq_length 510 \
# --gpus 1 \
# --task nela \
# --max_grad_norm 1 \
# --num_train_epochs 3 \
# --train_path "data/NELA/train_4.tsv" \
# --val_path "data/NELA/test_4.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \
#
#
# python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_nela_5 \
# --train_batch_size 1 \
# --eval_batch_size 1 \
# --learning_rate 2e-5 \
# --num_labels 3 \
# --max_seq_length 510 \
# --gpus 1 \
# --task nela \
# --max_grad_norm 1 \
# --num_train_epochs 3 \
# --train_path "data/NELA/train_5.tsv" \
# --val_path "data/NELA/test_5.tsv" \
# --seed 42 \
# --do_predict \
# --do_train \

python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_history_links_nowiki \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 128 \
--gpus 1 \
--task history_links_nowiki \
--max_grad_norm 1 \
--num_train_epochs 1 \
--train_path "data/train.tsv" \
--val_path "data/val.tsv" \
--history_train_path "data/semantic_train_results.tsv" \
--history_val_path "data/semantic_val_results.tsv" \
--link_train_path "data/train_links_processed.json" \
--link_val_path "data/val_links_processed.json" \
--seed 42 \
--do_predict \
--do_train \
