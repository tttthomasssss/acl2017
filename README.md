## Dependencies
	
The code relies on several 3rd party libraries:

* numpy
* scipy
* scikit-learn
* dill
* joblib
* nltk
* sqlitedict
	
In addition the following code (which has its own dependencies) is necessary for performing distributional inference:

* DiscoUtils: `https://github.com/tttthomasssss/DiscoUtils`

----

## Installation

Apart from `DiscoUtils` which needs to be installed manually, all requirements as well as the codebase itself can be installed with:
	
	cd path/to/apt-toolkit
	pip install -e .

----

## Resources

Vectors from the paper `Improving Semantic Composition with Offset Inference` will be posted on github - unfortunately they were too large for the ACL submission system.

----

## Usage

#### Loading vectors:

	from apt_toolkit.utils import vector_utils
	
	vectors = vector_utils.load_vector_cache('path/to/vectors', filetype='dill') # Loads the higher-order dependency-typed vectors as a `dict` of `dicts`
	
#### Creating Offset Representations:
	
	from apt_toolkit.composition import mozart
	from apt_toolkit.distributional_inference import distributional_inference
	from apt_toolkit.utils import vector_utils
	
	from discoutils.thesaurus_loader import Vectors
	
	'''
	Loads a standard set of vectorised APTs, creates some offsets and adds them back to the lexicon
	This is one way - and perhaps the simplest way - of doing creating neighbours for offsets
	'''
	
	# Load Vectors
	vectors = vector_utils.load_vector_cache('path/to/vectors', filetype='dill')
	
	noun_vector = vectors['quantity']
	adj_vector = vectors['large']
	
	# Offset and add the resulting representation back to the lexicon 
	offset_vector_adj = vector_utils.create_offset_vector(adj_vector, 'amod')
	vectors['__OFFSET_amod_large'] = offset_vector_adj
	
	# Can also offset the noun vector or any other vectors and add them back to the lexicon
	offset_vector_noun = vector_utils.create_offset_vector(noun_vector, '!amod')
	vectors['__OFFSET_!amod_quantity'] = offset_vector_noun
	
	# Store vectors back to file
	vector_utils.save_vector_cache(vectors, 'path/to/offsets', filetype='dill')
	
#### Composing Vectors with Offset Inference:
	
	from apt_toolkit.distributional_inference import distributional_inference
	from apt_toolkit.utils import vector_utils
	
	# Load Vectors with previously added offset representations
	vectors = vector_utils.load_vector_cache('path/to/offsets', filetype='dill')
	
	adj_vector = vectors['exciting']
	
	# Use offset Inference to enrich the noun-view of the adjective "exciting"
	exciting_offset = vector_utils.create_offset_vector(adj_vector, 'amod')
	rich_exciting_offset = distributional_inference.static_top_n(vectors=vectors, words=['__OFFSET_amod_exciting'], num_neighbours=20)
	
	# Use the standard DI algorithm to infer unobserved co-occurrence features for the noun "book"
	rich_book = distributional_inference.static_top_n(vectors=vectors, words=['book'], num_neighbours=20)
	
	# Now the two vectors can be composed
	composed_vector = mozart.intersect_apts(rich_exciting_offset, rich_book)