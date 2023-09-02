# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import errno
import json
import os
import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.mkdir(directory) #如果文件不存在，尝试创建
        except OSError as e: #OSError(文件名中存在敏感字符）
            if e.errno != errno.EEXIST: #如果不是不存在，则raise
                raise

def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
