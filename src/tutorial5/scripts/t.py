#!/usr/bin/env python
import numpy as np
from enum import Enum
import random
from sklearn import tree
import csv
import numpy as np
from pathlib import Path
import csv

my_file = Path("./file_name.csv")
if my_file.is_file():
    with open('file_name.csv', 'rb') as f:
        reader = csv.reader(f)
        rows = [[int(row[0]), int(row[1]), int(row[2])] for row in reader if row]
        print(rows)