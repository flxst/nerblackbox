if [ -z "$1" ]; then
  echo "No argument supplied"
  exit 1
fi

echo "NER dataset:" $1

if [ "$1" = 'swedish_ner_corpus' ]; then
  git clone https://github.com/klintan/swedish-ner-corpus.git datasets/ner/swedish_ner_corpus
elif [ "$1" = 'SUC' ]; then
  echo
else
  echo "NER dataset" $ner_dataset "unknown!"
  exit 1
fi

python datasets/script_create_ner_label_mapping_json_for_ner_processor.py --ner_dataset $1 --modify
python datasets/script_format_data.py --ner_dataset $1
python datasets/script_analyze_data.py --ner_dataset $1
