#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_links_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_history_links_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv" \
#--val_path "data/val.tsv" \
#--history_val_path "data/semantic_val_results.tsv" \
#--link_val_path "data/val_links_processed.json"
#--do_train \
#--train_path "data/train.tsv" \
#--history_train_path "data/semantic_train_results.tsv" \
#--link_train_path "data/train_links_processed.json" \


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_history_links_style \
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
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv" \
#--do_predict \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv" \

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_history_links_style \
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
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_history_links_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_links_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_history \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 42 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv"
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_history \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_history \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--test_path "data/test.tsv" \
#--link_test_path "data/test_links_processed.json" \
#--history_test_path "data/semantic_test_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_history \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 36 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv"
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_history \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv"
#
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
#--seed 0 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv"


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_history_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task history_style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 36 \
#--test_path "data/test.tsv" \
#--history_test_path "data/semantic_test_results.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results_links \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task links \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--do_predict \
#--seed 42 \
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv" \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \


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
#--do_predict \
#--seed 42 \
#--val_path "data/val.tsv" \
#--link_val_path "data/val_links_processed.json" \
#--history_val_path "data/semantic_val_results.tsv" \
#--do_train \
#--train_path "data/train.tsv" \
#--link_train_path "data/train_links_processed.json" \
#--history_train_path "data/semantic_train_results.tsv" \
#--test_path "data/val.tsv" \
#--link_test_path "data/val_links_processed.json" \
#--history_test_path "data/semantic_val_results.tsv"


#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 36 \
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv" \
#--test_path "data/test.tsv"

#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 42 \
#--do_train \
#--train_path "data/train.tsv" \
#--val_path "data/val.tsv" \
#--test_path "data/test.tsv"

python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 128 \
--gpus 1 \
--task style \
--max_grad_norm 1 \
--num_train_epochs 3 \
--seed 0 \
--test_path "data/val.tsv"

python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 128 \
--gpus 1 \
--task style \
--max_grad_norm 1 \
--num_train_epochs 3 \
--seed 42 \
--test_path "data/val.tsv"

python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 36/results_style \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 128 \
--gpus 1 \
--task style \
--max_grad_norm 1 \
--num_train_epochs 3 \
--seed 36 \
--test_path "data/val.tsv"

#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 42/results_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 42 \
#--test_path "data/test.tsv"
#
#python -m src.clf.trainer --num_labels 2 --model_name_or_path bert-base-uncased --output_dir 0/results_style \
#--train_batch_size 1 \
#--eval_batch_size 1 \
#--learning_rate 2e-5 \
#--num_labels 2 \
#--max_seq_length 128 \
#--gpus 1 \
#--task style \
#--max_grad_norm 1 \
#--num_train_epochs 3 \
#--seed 0 \
#--test_path "data/test.tsv"