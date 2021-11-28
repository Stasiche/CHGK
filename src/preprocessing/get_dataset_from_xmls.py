
import xml.etree.ElementTree as ET
import csv

import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import deque

from tqdm import tqdm
from typing import Dict


def filter_text(inp: str) -> str:
    return inp.replace('\n', ' ') if inp else None

def reset_dict() -> Dict:
    return {'question': None,
            'answer': None,
            'comments': None
            }

def get_parsed_el(url: str) -> Dict:
    tree = ET.ElementTree(file=urlopen(url))
    root = tree.getroot()
    for child in root:
        if child.tag == 'question':
            res = reset_dict()
            for el in child:
                if el.tag.lower() in res.keys():
                    res[el.tag.lower()] = filter_text(el.text)

            yield res


main_url = 'http://questions.chgk.info'
fieldnames = ['question', 'answer', 'comments']

with open('../../data/questions_list.txt', 'r') as f:
    ql = set(map(lambda x: x.strip('\n'), f.readlines()))

with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for l in tqdm(ql):
        l = l.replace('-q.html', '-a.html')
        page = requests.get(f'{main_url}{l}')
        soup = BeautifulSoup(page.content, 'html5lib')
        db_url = [el.get('href') for el in soup.find_all('a') if el.get('href') and el.get('href').startswith('/dbxml.php')]
        if len(db_url) > 1:
            print(f'!!! problems. {l}\n{db_url}\n_________')
        else:
            for dataset_line in get_parsed_el(f'{main_url}{db_url[0]}'):
                writer.writerow(dataset_line)



