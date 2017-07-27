__author__ = 'thk22'
import array
import collections
import logging
import math

from scipy import sparse
from sqlitedict import SqliteDict
import dill
import numpy as np
import joblib


def split_path_from_word(event, universal_deps=[]):
    if (event.count(':') > 1):
        path = []
        word = []

        for dep in reversed(event.split('\xbb')):
            d = dep if not dep.startswith('_') else dep[1:]

            if (':' in d):
                subpath = []
                for sub_dep in reversed(dep.split(':')):
                    sd = sub_dep if not sub_dep.startswith('_') else sub_dep[1:]
                    if (sd in universal_deps):
                        subpath.insert(0, sub_dep)
                    else:
                        word.insert(0, sub_dep)

                # Handle words that happen to have the same signifier as a dependency
                if (len(word) <= 0):
                    word.insert(0, subpath.pop())

                path.insert(0, ':'.join(subpath))
            else:
                path.insert(0, dep)
        return '\xbb'.join(path), ':'.join(word)
    elif (event.count(':') <= 0): # Could be a simple co-occurrence count vector
        return '', event
    else:
        return event.rsplit(':', 1)


def apply_offset_path(path, offset, incompatible_paths='strict'):
    '''
    :param path: dependency path feature
    :param offset: offset path
    :param incompatible_paths: pass 'strict' to exclude incompatible paths (double negative path) or 'ignore' to leave them in
    :return: offset_path or None if the path is incompatible and `incompatible_paths='strict'`
    '''

    if (path.startswith(':')): # EPSILON
        offset_path = offset + path
    else:
        head, feat = split_path_from_word(path)

        parts = head.split('\xbb')

        if ('_{}'.format(parts[0]) == offset or '_{}'.format(offset) == parts[0]):
            offset_path = '{}:{}'.format('\xbb'.join(parts[1:]), feat)
        elif (path.startswith('_') and incompatible_paths == 'strict'): # Incompatible feature
            offset_path = None
        else:
            offset_path = '\xbb'.join([offset, path])

    return offset_path


def create_offset_vector(vector, offset_path, incompatible_paths='strict'):
    # Translate from my notation to Dave's notation
    if (offset_path.startswith('!')):
        offset_path = '_' + offset_path[1:]

    v = {}
    for feat in vector.keys():
        new_feat_path = apply_offset_path(feat, offset_path, incompatible_paths=incompatible_paths)

        if (new_feat_path is not None):
            v[new_feat_path] = vector[feat]

    return v


def collapse_offset_vector(offset_vector, offset_path=None):
    reduced_offset_vector = collections.defaultdict(float)
    for key in offset_vector.keys():
        head, feat = key.rsplit(':', 1)

        parts = head.split('\xbb')
        if (len(parts) > 1):
            offset = parts[0] if offset_path is None else offset_path
            if (offset[1:] == parts[1] or offset == parts[1][1:]):
                new_key = '{}:{}'.format('\xbb'.join(parts[2:]), feat)
            else:
                new_key = key
        else:
            new_key = key

        reduced_offset_vector[new_key] += offset_vector[key]

        return reduced_offset_vector


def load_vector_cache(vector_in_file, filetype='', **kwargs):
    if (vector_in_file.endswith('.dill') or filetype == 'dill'):
        with open(vector_in_file, 'rb') as data_file:
            vectors = dill.load(data_file)
        return vectors
    elif (vector_in_file.endswith('.joblib') or filetype == 'joblib'):
        return joblib.load(vector_in_file)
    elif (vector_in_file.endswith('.sqlite') or filetype == 'sqlite'):
        return SqliteDict(vector_in_file, autocommit=kwargs.pop('autocommit', True), flag='r')
    else:
        raise NotImplementedError


def save_vector_cache(vectors, vector_out_file, filetype='', **kwargs):
    logging.info("Saving {} vectors to cache {}".format(len(vectors),vector_out_file))
    if (vector_out_file.endswith('.dill') or filetype == 'dill'):
        with open(vector_out_file, 'wb') as data_file:
            dill.dump(vectors, data_file, protocol=kwargs.get('dill_protocol', 3))
    elif (vector_out_file.endswith('.joblib') or filetype == 'joblib'):
        joblib.dump(vectors, vector_out_file, compress=kwargs.get('joblib_compression', 3),
                    protocol=kwargs.get('joblib_protocol', 3))
    elif (vector_out_file.endswith('.sqlite') or filetype == 'sqlite'):
        autocommit = kwargs.pop('autocommit', True)
        if (isinstance(vectors, SqliteDict)):
            vectors.commit()
        else:
            with SqliteDict(vector_out_file, autocommit=autocommit) as data_file:
                for key, value in vectors.items():
                    data_file[key] = value

                if (not autocommit):
                    data_file.commit()
    else:
        raise NotImplementedError