#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NER数据集持久化，与标注文件之间的互转。

import os, json, re
from dataclasses import dataclass, field
from typing import List

from loguru import logger
from tqdm import tqdm
