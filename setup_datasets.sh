########################################################################################################################
if [ -z "$1" ]; then
  echo "No argument supplied"
  exit 1
fi

if [[ "$1" != 'swedish_ner_corpus' && "$1" != 'SUC' ]]; then
  echo "NER dataset" $ner_dataset "unknown!"
  exit 1
fi

echo "NER dataset:" $1

########################################################################################################################
python datasets/script_get_data.py --ner_dataset $1
python datasets/script_create_ner_tag_mapping.py --ner_dataset $1 --modify
python datasets/script_format_data.py --ner_dataset $1
python datasets/script_analyze_data.py --ner_dataset $1 --verbose
