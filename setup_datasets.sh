git clone https://github.com/klintan/swedish-ner-corpus.git datasets/ner/swedish_ner_corpus
python datasets/script_create_ner_label_mapping_json_for_ner_processor.py --ner_dataset swedish_ner_corpus --modify
python datasets/script_format_data.py --ner_dataset swedish_ner_corpus
python datasets/script_analyze_data.py --ner_dataset swedish_ner_corpus
