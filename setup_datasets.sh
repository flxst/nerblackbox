git clone https://github.com/klintan/swedish-ner-corpus.git datasets/ner/swedish_ner_corpus
python datasets/format_swedish_ner_corpus.py
python datasets/create_ner_label_mapping_json_for_ner_processor.py --model swedish_ner_corpus