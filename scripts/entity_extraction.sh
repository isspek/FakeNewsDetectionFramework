#python -m src.elasticsearch.extract_bio_entities --data constraint --input_file data/val.tsv --output_file constraint_val_entities.json
#python -m src.elasticsearch.extract_bio_entities --data constraint --input_file data/train.tsv --output_file constraint_train_entities.json
#python -m src.elasticsearch.extract_bio_entities --data constraint --input_file data/test.tsv --output_file constraint_test_entities.json

#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/36/val.tsv --output_file data/coaid/article/36/coaid_val_entities.json
#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/36/train.tsv --output_file data/coaid/article/36/coaid_train_entities.json
#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/36/test.tsv --output_file data/coaid/article/36/coaid_test_entities.json
#
#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/42/val.tsv --output_file data/coaid/article/42/coaid_val_entities.json
#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/42/train.tsv --output_file data/coaid/article/42/coaid_train_entities.json
#python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/article/42/test.tsv --output_file data/coaid/article/42/coaid_test_entities.json

#python -m src.pk.extract_bio_entities --data recovery --input_file data/recovery/article/val.tsv --output_file data/recovery/article/recovery_val_entities.json
#python -m src.pk.extract_bio_entities --data recovery --input_file data/recovery/article/train.tsv --output_file data/recovery/article/recovery_train_entities.json
#python -m src.pk.extract_bio_entities --data recovery --input_file data/recovery/article/test.tsv --output_file data/recovery/article/recovery_test_entities.json

python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/0/val.tsv --output_file data/coaid/claims/0/coaid_val_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/0/train.tsv --output_file data/coaid/claims/0/coaid_train_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/0/test.tsv --output_file data/coaid/claims/0/coaid_test_entities.json


python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/36/val.tsv --output_file data/coaid/claims/36/coaid_val_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/36/train.tsv --output_file data/coaid/claims/36/coaid_train_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/36/test.tsv --output_file data/coaid/claims/36/coaid_test_entities.json

python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/42/val.tsv --output_file data/coaid/claims/42/coaid_val_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/42/train.tsv --output_file data/coaid/claims/42/coaid_train_entities.json
python -m src.pk.extract_bio_entities --data coaid --input_file data/coaid/claims/42/test.tsv --output_file data/coaid/claims/42/coaid_test_entities.json