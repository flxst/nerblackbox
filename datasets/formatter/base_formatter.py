
class BaseFormatter:

    def __init__(self, ner_label_list):
        self.ner_label_list = ner_label_list

    def create_ner_label_mapping(self, with_tags: bool, modify: bool):
        """
        create customized ner label mapping to map labels in original data to labels in formatted data
        ----------------------------------------------------------------------------------------------
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :param modify:    [bool], if True: modify labels as specified in method modify_ner_label_mapping()
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        # full label list
        if with_tags:
            _label_lists_extended = [[f'B-{label}', f'I-{label}'] for label in self.ner_label_list]
            label_list_full = ['O'] + [l_i for l in _label_lists_extended for l_i in l]
        else:
            label_list_full = ['O'] + self.ner_label_list

        # map each label to itself
        ner_label_mapping_original = {k: k for k in label_list_full}

        if modify:
            return self.modify_ner_label_mapping(ner_label_mapping_original, with_tags=with_tags)
        else:
            return ner_label_mapping_original

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def modify_ner_label_mapping(self, ner_label_mapping_original, with_tags: bool):
        """
        customize ner label mapping if wanted
        -------------------------------------
        :param ner_label_mapping_original: [dict] w/ keys = labels in original data, values = labels in original data
        :param with_tags: [bool], if True: create labels with BIO tags, e.g. 'B-PER', 'I-PER', 'B-LOC', ..
                                  if False: create simple labels, e.g. 'PER', 'LOC', ..
        :return: ner_label_mapping: [dict] w/ keys = labels in original data, values = labels in formatted data
        """
        pass

    @staticmethod
    def read_original_file(phase):
        pass

    @staticmethod
    def write_formatted_csv(phase, rows, dataset_path):
        pass
