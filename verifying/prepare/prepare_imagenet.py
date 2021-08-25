import re
import sys
import json
import logging
import pathlib2
import pandas as pd
from tabulate import tabulate
from typing import Union, List, Set, Dict


# info filenames
CLASS_TO_LABEL_ID_JSON = "class_to_label_id.json"
LABEL_ID_TO_LABELS_JSON = "label_id_to_labels.json"
INDEX_TO_LABELS_JSON = "index_to_labels.json"
INDEX_TO_LABEL_ID_JSON = "index_to_label_id.json"
IMAGE_TO_LABEL_ID_CSV = "image_to_label_id.csv"
INET_VAL_LIST_FILENAME = "inet.val.list"
SYNSET_WORDS_TXT = "synset_words.txt"
MSCOCO_TO_IMAGENET_CATEGORY_MAPPING_TXT = 'MSCOCO_to_ImageNet_category_mapping.txt'

# set up path
__root__ = pathlib2.Path(__file__).absolute().parent
project_root = __root__.parent
data_dir = project_root / 'data'
imagenet_data_dir = data_dir / 'imagenet'
dataset_info_dir = imagenet_data_dir / "info"

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

RUN_ALL_MODE = 0
CROSS_MAPPING_MODE = 1
CLASS_TO_LABEL_ID = 2
IMAGE_TO_LABEL_ID = 3


def main(modes: Union[List, Set, int] = RUN_ALL_MODE):
    """main function (entrypoint)
    Decide which files to generate

    :param modes: mode or modes, indicating which files to generate_bootstrap_payload, defaults to RUN_ALL_MODE
    :type modes: Union[List, Set, int], optional
    """
    logger.info(f"begin {__file__}")
    if not dataset_info_dir.exists():
        logger.info("create 'dataset info' directory")
        dataset_info_dir.mkdir(parents=True, exist_ok=True)
    if (not isinstance(modes, int) and CROSS_MAPPING_MODE in modes) or \
            modes == RUN_ALL_MODE or modes == CROSS_MAPPING_MODE:
        gen_cross_mapping_json()
    if (not isinstance(modes, int) and CLASS_TO_LABEL_ID in modes) or \
            modes == RUN_ALL_MODE or modes == CLASS_TO_LABEL_ID:
        gen_class_to_label_id()
    if (not isinstance(modes, int) and IMAGE_TO_LABEL_ID in modes) or \
            modes == RUN_ALL_MODE or modes == IMAGE_TO_LABEL_ID:
        index_to_label_id = gen_cross_mapping_json()
        df = gen_image_to_label_id_csv(index_to_label_id)
        labels_csv_df = gen_labels_csv(df.reset_index())
        labels_csv_df.to_csv(imagenet_data_dir / 'imgs' / 'labels.csv')

def gen_labels_csv(df: pd.DataFrame) -> pd.DataFrame:

    return pd.DataFrame(data={'filename': list(df['image_name'].apply(lambda x: x + ".JPEG")), 'label': list(df['label_index'])})


def gen_image_to_label_id_csv(index_to_label_id: Dict[int, str]):
    """Generate a csv file that's responsible for image-to-label_id mapping

    :param index_to_label_id: index_to_label_id dictionary, a dependency
    :type index_to_label_id: Dict[int, str]
    """
    # image name to label id and index
    logger.info(f"generate {IMAGE_TO_LABEL_ID_CSV}")
    with open(str(imagenet_data_dir / INET_VAL_LIST_FILENAME), "r") as f:
        lines = f.readlines()
    label_id_to_index = {v: k for k, v in index_to_label_id.items()}
    data = [list(re.findall(r"(ILSVRC2012\_val\_\d+)\.(n\d+)", line)[0]) for line in lines]
    for row in data:
        row.append(label_id_to_index[row[1]])
    df = pd.DataFrame(data, columns=["image_name", "label_id", "label_index"])
    df.set_index("image_name", inplace=True)
    df.to_csv(str(dataset_info_dir / IMAGE_TO_LABEL_ID_CSV))
    print(tabulate(df.head(), headers='keys', tablefmt='pretty'))
    return df

def gen_class_to_label_id() -> None:
    """generate json file containing class-to-label_id mapping
    """
    # class to label id
    logger.info(f"generate {CLASS_TO_LABEL_ID_JSON}")
    class_to_label_id_dict = {}
    with open(str(imagenet_data_dir / MSCOCO_TO_IMAGENET_CATEGORY_MAPPING_TXT), "r") as f:
        lines = f.readlines()
    valid_lines = list(filter(lambda l: '#' != l[0], lines))  # remove comment lines with #
    clean_line = ''.join(valid_lines).replace('\n', '').replace(' ', '')  # remove new line character and spaces
    parsed_lines = [match[0] for match in re.findall(r'([a-z]+=\[(n\d+,)*n\d+])', clean_line)]
    for line in parsed_lines:
        class_, ids, _ = re.findall(r'([a-z]+)=\[((n\d+,)*n\d+)]', line)[0]
        class_to_label_id_dict[class_] = ids.split(',')
    with open(str(dataset_info_dir / CLASS_TO_LABEL_ID_JSON), "w") as f:
        json.dump(class_to_label_id_dict, f, indent=2)


def gen_cross_mapping_json() -> Dict[int, str]:
    """Generate 3 json files for different types of mapping:
    - label id to labels
        - id in str to words
    - index to labels
        - int to words
    - index to label id
        - int to str

    :return: index_to_label_id dictionary, returned because this will be a dependency of another function
    :rtype: Dict[int, str]
    """
    logger.info("constructing mapping dictionary/json")
    label_id_to_labels, index_to_labels, index_to_label_id = {}, {}, {}
    with open(str(imagenet_data_dir / SYNSET_WORDS_TXT), "r") as f:
        for i, line in enumerate(f.readlines()):
            match = re.findall(r"(n\d+) (.*)", line)
            if match[0][0] not in label_id_to_labels.keys():
                label_id_to_labels[match[0][0]] = []
            index_to_label_id[i] = match[0][0]
            label_id_to_labels[match[0][0]] += match[0][1].split(", ")
            index_to_labels[i] = match[0][1].split(", ")
    with open(str(dataset_info_dir / LABEL_ID_TO_LABELS_JSON), "w") as f:
        json.dump(label_id_to_labels, f, indent=2)
    with open(str(dataset_info_dir / INDEX_TO_LABELS_JSON), "w") as f:
        json.dump(index_to_labels, f, indent=2)
    with open(str(dataset_info_dir / INDEX_TO_LABEL_ID_JSON), "w") as f:
        json.dump(index_to_label_id, f, indent=2)
    return index_to_label_id


if __name__ == "__main__":
    main()
