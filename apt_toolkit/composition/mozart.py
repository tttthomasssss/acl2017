__author__ = 'thk22'
import operator


def union_apts(vector_1, vector_2, merge_fn=operator.add):
	"""
	Compose 2 APT vectors (as `dicts`) by the union of their distributional features.

	:param vector_1: constituent vector 1
	:param vector_2: constituent vector 2
	:param merge_fn: any function to combine the two values (e.g. min, max, operator.add, operator.mul), default is operator.add
	:return: composed APT (as `dict`)
	"""
	composed_vector = {}
	for feat in (set(vector_1.keys()) | set(vector_2.keys())):
		composed_vector[feat] = merge_fn(vector_1.get(feat, 0.), vector_2.get(feat, 0.))

	return composed_vector


def intersect_apts(vector_1, vector_2, merge_fn=operator.add):
	"""
	Compose 2 APT vectors (as `dicts`) by the intersection of their distributional features.

	:param vector_1: constituent vector 1
	:param vector_2: constituent vector 2
	:param merge_fn: any function to combine the two values (e.g. min, max, operator.add, operator.mul), default is operator.add
	:return: composed APT (as `dict`)
	"""
	composed_vector = {}
	for feat in (set(vector_1.keys()) & set(vector_2.keys())):
		composed_vector[feat] = merge_fn(vector_1[feat], vector_2[feat])

	return composed_vector