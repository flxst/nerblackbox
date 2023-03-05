LABEL_COLORS = [
    "#000000",  # black
    "#196F3D",  # green dark
    "#1F618D",  # blue dark
    "#FF0000",  # red
    "#A52A2A",  # brown
    "#FFC300",  # orange
    "#566573",  # gray dark
    "#AED6F1",  # blue light
    "#922B21",  # red dark
    "#7DCEA0",  # green light
    "#873600",  # brown dark
    "#0000FF",  # blue
    "#ABB2B9",  # gray light
    "#008000",  # green
    "#BA4A00",  # brown light
    "#F1948A",  # red light
    "#0D98BA",  # blue green
    "#F4A460",  # brown sandy
]


def get_label_color(index: int) -> str:
    """
    get label color for index

    Args:
        index: e.g. 3

    Returns:
        label_color: e.g. "#FF0000"

    """
    effective_index = index % len(LABEL_COLORS)
    return LABEL_COLORS[effective_index]
