__author__ = 'thk22'

from discoutils.thesaurus_loader import Vectors
from nltk.corpus import wordnet
from scipy import sparse
import numpy as np


def static_top_n(vectors, words=None, num_neighbours=10, alpha='auto', nn_metric='cosine', **kwargs):
	"""
	Perform smoothing by associative inference
	:param vectors: Original elementary APTs
	:param words: Lexemes of interest to apply distributional inference on (pass None for all lexemes)
	:param num_neighbours: Number of neighbours used for distributional inference
	:param alpha: weighting of original vector (default='auto', which multiplies the original vectors by `num_neighbours`)
	:param nn_metric: nearest neighbour metric to use (default='cosine'; supported are 'cosine' and 'euclidean')
	:return: smoothed apt vector
	"""
	smoothed_vectors = {}
	if (isinstance(vectors, Vectors)):
		disco_vectors = vectors
	else: # Passive-Aggressive-Defensive loading cascade
		if (isinstance(vectors, dict)):
			disco_vectors = Vectors.from_dict_of_dicts(vectors)
		else:
			raise ValueError('Unsupported type[{}] for `vectors` supplied. Supported types are [`discoutils.thesaurus_loader.Vectors` and `dict`]!')

	if (not kwargs.pop('is_initialised', False)):
		disco_vectors.init_sims(n_neighbors=num_neighbours, nn_metric=nn_metric, knn='brute' if nn_metric == 'cosine' else 'kd_tree')

	words = words if words is not None else vectors.keys()

	a = alpha if alpha != 'auto' else num_neighbours
	for w in words:
		if (w not in disco_vectors):
			smoothed_vectors[w] = sparse.csr_matrix((1, disco_vectors.matrix.shape[1]))
			continue

		neighbours = []
		try:
			neighbours = disco_vectors.get_nearest_neighbours(w)
		except ValueError as ex:
			import logging
			logging.error('Failed to retrieve neighbours for w={}: {}...'.format(w, ex))
			raise ex

		# Enrich original vector
		apt = disco_vectors.get_vector(w)
		if (apt is None): # OOV
			apt = sparse.csr_matrix((1, disco_vectors.matrix.shape[1]))
		apt *= a

		for neighbour, _ in neighbours:
			apt += disco_vectors.get_vector(neighbour)

		smoothed_vectors[w] = apt.copy()

	return disco_vectors, smoothed_vectors


def density_window(vectors, words=None, num_neighbours=10, window_size=0.1, alpha='auto', nn_metric='cosine', **kwargs):
	"""
	Perform smoothing by associative inference
	:param vectors: Original elementary APTs
	:param words: Lexemes of interest to apply distributional inference on (pass None for all lexemes)
	:param num_neighbours: Maximum number of neighbours used for distributional inference
	:param window_size: proportional distance to nearest neighbour, defining the parzen window for each vector individually (default=0.1)
	:param alpha: weighting of original vector (default='auto', which multiplies the original vectors by `num_neighbours`)
	:param nn_metric: nearest neighbour metric to use (default='cosine'; supported are 'cosine' and 'euclidean')
	:return: smoothed apt vector
	"""
	smoothed_vectors = {}
	if (isinstance(vectors, Vectors)):
		disco_vectors = vectors
	else: # Passive-Aggressive-Defensive loading cascade
		if (isinstance(vectors, dict)):
			disco_vectors = Vectors.from_dict_of_dicts(vectors)
		else:
			raise ValueError('Unsupported type[{}] for `vectors` supplied. Supported types are [`discoutils.thesaurus_loader.Vectors` and `dict`]!')

	if (not kwargs.pop('is_initialised', False)):
		disco_vectors.init_sims(n_neighbors=num_neighbours, nn_metric=nn_metric, knn='brute' if nn_metric == 'cosine' else 'kd_tree')

	words = words if words is not None else vectors.keys()

	a = alpha if alpha != 'auto' else num_neighbours
	for w in words:
		if (w not in disco_vectors): continue
		# Retrieve top neighbour
		top_neighbour = disco_vectors.get_nearest_neighbours(w)[0]

		# Anything within `distance_threshold` is still considered for inference
		distance_threshold = top_neighbour[1] * (1+window_size)

		neighbours = []
		for neighbour, distance in disco_vectors.get_nearest_neighbours(w):
			if (distance > distance_threshold): break

			neighbours.append((neighbour, distance))

		# Enrich original vector
		apt = disco_vectors.get_vector(w) * a

		for neighbour, _ in neighbours:
			apt += disco_vectors.get_vector(neighbour)

		smoothed_vectors[w] = apt.copy()

	return disco_vectors, smoothed_vectors


def wordnet_synsets(vectors, words, num_neighbours, alpha='auto', nn_metric='cosine', **kwargs):
	"""
	Perform smoothing by associative inference
	:param vectors: Original elementary APTs
	:param words: Lexemes of interest to apply distributional inference on (pass None for all lexemes), !!!Need to be (word, pos) tuples!!!
	:param num_neighbours: Maximum number of neighbours used for distributional inference
	:param alpha: weighting of original vector (default='auto', which multiplies the original vectors by `num_neighbours`)
	:param nn_metric: nearest neighbour metric to use (default='cosine'; supported are 'cosine' and 'euclidean')
	:return: smoothed apt vector
	"""
	smoothed_vectors = {}
	if (isinstance(vectors, Vectors)):
		disco_vectors = vectors
	else: # Passive-Aggressive-Defensive loading cascade
		if (isinstance(vectors, dict)):
			disco_vectors = Vectors.from_dict_of_dicts(vectors)
		else:
			raise ValueError('Unsupported type[{}] for `vectors` supplied. Supported types are [`discoutils.thesaurus_loader.Vectors` and `dict`]!')

	if (not kwargs.pop('is_initialised', False)):
		disco_vectors.init_sims(n_neighbors=num_neighbours, nn_metric=nn_metric, knn='brute' if nn_metric == 'cosine' else 'kd_tree')

	words = words if words is not None else vectors.keys()

	a = alpha if alpha != 'auto' else num_neighbours
	for w, pos in words:
		if (w not in disco_vectors): continue
		neighbours = set()
		for syn in wordnet.synsets(w, pos=pos):
			n = syn.name().split('.')[0]
			if (n != w):
				neighbours.add(n)

		# Get indices of neighbours
		idx = []
		for i, n in enumerate(neighbours, 1):
			if (i > num_neighbours): break
			if (n in disco_vectors):
				idx.append(disco_vectors.name2row[n])

		A = disco_vectors.matrix[np.array(idx)]

		# Retrieve vector for `w` and add `A` to it and apply alpha weighting to original APT
		apt = sparse.csr_matrix(disco_vectors.get_vector(w).multiply(a) + A.sum(axis=0)) # Should still be sparse enough

		smoothed_vectors[w] = apt.copy()

	return disco_vectors, smoothed_vectors