from typing import List
import pandas as pd


class Annotation:
    """
    Attributes:
        classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        scheme:                                  e.g. "plain" or "bio"
    """

    def __init__(self, classes: List[str]):
        """
        Args:
            classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        """
        self.classes: List[str] = classes
        self.scheme: str = ""
        self._process()

    def _process(self):
        """
        changed attributes:
            scheme: [str], e.g. "plain" or "bio"
            classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        """
        self._get_scheme()  # attr: scheme
        if self.scheme == "bio":
            self._ensure_completeness_in_case_of_bio_tags()  # attr: classes
        elif self.scheme == "bilou":
            self._ensure_completeness_in_case_of_bilou_tags()  # attr: classes
        self._sort_classes()  # attr: classes

    def _get_scheme(self) -> None:
        """
        derive annotation scheme from classes

        changed attributes:
            scheme: [str], e.g. "plain" or "bio"
        """
        if any(["-" in tag for tag in self.classes]):
            if any(["L-" in tag for tag in self.classes]) or any(
                ["U-" in tag for tag in self.classes]
            ):
                self.scheme = "bilou"
            else:
                self.scheme = "bio"
        else:
            self.scheme = "plain"

    def _ensure_completeness_in_case_of_bio_tags(self) -> None:
        """
        in case of BIO-tags: make sure that there is an "I-*" tag for every "B-*" tag

        changed attributes:
            classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        """
        b_tags = [tag for tag in self.classes if tag.startswith("B")]
        for b_tag in b_tags:
            i_tag = b_tag.replace("B-", "I-")
            if i_tag not in self.classes:
                self.classes.append(i_tag)

    def _ensure_completeness_in_case_of_bilou_tags(self) -> None:
        """
        in case of BILOU-tags: make sure that there is an "I-*", "L-*", "U-" tag for every "B-*" tag

        changed attributes:
            classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        """
        b_tags = [tag for tag in self.classes if tag.startswith("B")]
        for b_tag in b_tags:
            for x in ["I-", "L-", "U-"]:
                x_tag = b_tag.replace("B-", x)
                if x_tag not in self.classes:
                    self.classes.append(x_tag)

    def _sort_classes(self) -> None:
        """
        sort classes. in case of BIO-tags, all B-* tags come first.

        changed attributes:
            classes: tag classes present in dataset, e.g. ["PER", "ORG"] or ["B-person", "B-time", "I-person"]
        """
        self.classes = ["O"] + sorted([elem for elem in self.classes if elem != "O"])

    def change_scheme(self, new_scheme: str) -> "Annotation":
        if self.scheme == new_scheme:
            return self
        elif new_scheme == "plain":
            classes_bio_without_o = [elem for elem in self.classes if elem != "O"]
            classes_plain = ["O"] + list(
                pd.Series(classes_bio_without_o)
                .map(lambda x: x.split("-")[-1])
                .drop_duplicates()
            )

            annotation_plain = Annotation(classes_plain)
            return annotation_plain
        elif new_scheme in ["bio", "bilou"]:
            classes_plain_without_o = list(
                set([elem.split("-")[-1] for elem in self.classes if elem != "O"])
            )
            classes_bio_bilou = ["O"]
            for elem in classes_plain_without_o:
                if new_scheme == "bio":
                    classes_bio_bilou += [f"B-{elem}", f"I-{elem}"]
                else:  # bilou
                    classes_bio_bilou += [
                        f"B-{elem}",
                        f"I-{elem}",
                        f"L-{elem}",
                        f"U-{elem}",
                    ]

            annotation_bio_bilou = Annotation(classes_bio_bilou)
            return annotation_bio_bilou
        else:
            raise Exception(f"ERROR! new_scheme = {new_scheme} is not implemented.")
