# python -m src.style.lightning_classifier --pretrained bert-base-uncased --auto_select_gpus true --batch_size 8 --learning_rate 2e-5 --gradient_clip_val 1.0 --nela_train data/NELA/train_1.tsv --nela_test data/NELA/test_1.tsv --num_labels 3 --max_len 512 \
# --gpus 1 --max_epochs 1 --task constraint



python -m src.style.lightning_classifier --num_labels 2 --model_name_or_path bert-base-uncased --output_dir results \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--num_labels 2 \
--max_seq_length 128 \
--gpus 1 \
--task constraint \
--max_grad_norm 1 \
--num_train_epochs 1 \
--train_path "data/train.tsv" \
--val_path "data/val.tsv" \
--seed 42 \
--do_predict \
# --do_train \
