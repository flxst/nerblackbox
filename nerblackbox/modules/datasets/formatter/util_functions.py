import json


def get_ner_tag_mapping(path):
    """
    get ner_tag_mapping callable
    ----------------------------
    :param path: [str] to ner_tag_mapping json
    :return: [callable] ner_tag_mapping that applies ner_tag_mapping (loaded from json) or identical mapping
    """

    def without_bio(tag):
        """
        :param tag:                      [str], e.g. 'B-PER', 'ORG'
        :return: tag without bio prefix, [str], e.g.   'PER', 'ORG'
        """
        return tag.split("-")[-1]

    with open(path, "r") as f:
        ner_tag_mapping = json.load(f)
    return (
        lambda x: ner_tag_mapping[without_bio(x)]
        if without_bio(x) in ner_tag_mapping.keys()
        else x
    )
