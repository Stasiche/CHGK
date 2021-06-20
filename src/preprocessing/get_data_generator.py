import csv
from typing import Generator, OrderedDict


def get_data_generator(dataset_path: str) -> Generator[OrderedDict, None, None]:
    with open(dataset_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row
