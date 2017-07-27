__author__ = 'thk22'


def perform_distributional_inference(inference_function, disco_vectors, apt, word, num_neighbours=100, **kwargs):
	"""
	Wrapper for the DI functions in distributional_inference.py

	:param inference_function: callable of one of the functions in distributional_inference.py
	:param disco_vectors: vectors transformed by discoutils.thesaurus_loader.Vectors.from_dict_of_dicts()
	:param apt: higher-order dependency-typed vector as `dict`
	:param word: word to perform distributional inference for
	:param num_neighbours: number of neighbours
	:param kwargs: keyword arguments for any of the distributional inference functions
	:return:
	"""

	pos = kwargs.pop('pos')
	if ('wordnet' in inference_function.__name__):
		words = [(word, pos)]
	else:
		words = [word]

	_, smoothed_apt = inference_function(disco_vectors, words, num_neighbours=num_neighbours, **kwargs)

	if (isinstance(smoothed_apt[word], dict)):
		return smoothed_apt[word] # Inverse transform already performed
	else:
		smoothed_cols = smoothed_apt[word].nonzero()[1]

		for col in smoothed_cols:
			feat = disco_vectors.columns[col]

			apt[feat] = smoothed_apt[word][0, col]

		return apt