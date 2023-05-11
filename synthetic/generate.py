import math
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-points-per-concept", default=1000, type=int)
parser.add_argument("--modality-number", default=2, type=int)
parser.add_argument("--dominant-modality-concept-variance", default=0.5, type=float)
parser.add_argument("--other-modality-concept-variance", default=0.5, type=float)
parser.add_argument("--concept-number", default=32, type=int)
parser.add_argument("--concept-embedding-dimension", default=32, type=int)
parser.add_argument("--modality-embedding-dimension", default=32, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--use-hypercube-concepts", action='store_true')
parser.add_argument("--use-paired-concepts", action='store_true')
parser.add_argument("--individual-concepts-per-modality", default=0, type=int)
parser.add_argument("--paired-datum-number", default=5, type=int)
parser.add_argument('--out-path', default='datagen.pickle', type=str)
args = parser.parse_args()

def generate_binary(bit_count):
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')


    genbin(bit_count)
    return [bitstr_to_tuple(bs) for bs in binary_strings]

def generate_ohe(bit_count):
    binary_strings = []
    for i in range(bit_count):
      stuff = [0]*(bit_count)
      stuff[i] = 1
      binary_strings.append(tuple(stuff))
    return binary_strings


def bitstr_to_tuple(bitstr):
  return tuple(int(elem) for elem in bitstr)

# Number of data points
NUM_DATA = args.data_points_per_concept
# Number of samples from the same concept Gaussian ( i.e. 'modalities' )
NUM_MODALITIES = args.modality_number
# Standard deviation of generated Gaussian distributions
SEP =  args.other_modality_concept_variance # 0.5,1.0,2.0 with 100 concepts
# Number of concepts
NUM_CONCEPTS = args.concept_number # 2,10,100,1000 with 1.0 std here
# Number of paired data per concept
NUM_PAIRED_DATA = args.paired_datum_number

DIMENSION = args.modality_embedding_dimension
CONCEPT_DIM = args.concept_embedding_dimension


NUM_INDIV_CONCEPTS = args.individual_concepts_per_modality


# Generate Concept Means
if args.use_hypercube_concepts:
  internal_concept_dim = math.ceil(math.log(NUM_CONCEPTS, 2))
  concept_means = generate_binary(internal_concept_dim)
else:
  internal_concept_dim = NUM_CONCEPTS
  concept_means = generate_ohe(internal_concept_dim)


# Generate means of concepts, so that they're spaced out evenly a la sklearn.datasets.make_classification [
# I. Guyon, 'Design of experiments for the NIPS 2003 variable selection benchmark', 2003.]
keys = ['a','b','c','d','e']
TF_SEQ =  {keys[idx]:np.random.normal(0.0,1.0,(CONCEPT_DIM, DIMENSION)) for idx in range(NUM_MODALITIES)}

# Transform from concepts to concept_means
TFSTAR = np.random.uniform(-2.0,2.0,(internal_concept_dim, CONCEPT_DIM))




if not args.use_paired_concepts:
  total_raw_data = []
  count = 0
  for modality_idx in range(NUM_MODALITIES):
    temp_list = []
    for concept_idx in range(NUM_CONCEPTS):
      # For now, only change variance ( dominance ) in training
      sep_per_modality = args.dominant_modality_concept_variance if modality_idx == 0 else args.other_modality_concept_variance
      sample_set = np.random.multivariate_normal(concept_means[concept_idx],np.eye(internal_concept_dim)*sep_per_modality, (NUM_DATA,))
      if NUM_INDIV_CONCEPTS != 0 and concept_idx // NUM_INDIV_CONCEPTS != modality_idx and concept_idx < NUM_INDIV_CONCEPTS*NUM_MODALITIES:
        count += 1
        sample_set = np.concatenate((0.01*np.ones((NUM_DATA - NUM_PAIRED_DATA,internal_concept_dim)),np.random.multivariate_normal(concept_means[concept_idx],np.eye(internal_concept_dim)*sep_per_modality, (NUM_PAIRED_DATA,))),axis=0)
      temp_list.append(sample_set)
    total_raw_data.append(temp_list)
    print(count)
  X_train = np.array(total_raw_data).transpose((2,1,0,3)).reshape((-1,NUM_MODALITIES, internal_concept_dim))
  y_train = np.tile([i for i in range(NUM_CONCEPTS)], NUM_DATA)

  total_test_data = []

  for modality_idx in range(NUM_MODALITIES):
    temp_list = []
    for concept_idx in range(NUM_CONCEPTS):
      sample_set = np.random.multivariate_normal(concept_means[concept_idx],np.eye(internal_concept_dim)*SEP, (NUM_DATA,))
      temp_list.append(sample_set)
    total_test_data.append(temp_list)
  X_test = np.array(total_test_data).transpose((2,1,0,3)).reshape((-1,NUM_MODALITIES, internal_concept_dim))
  y_test = np.tile([i for i in range(NUM_CONCEPTS)], NUM_DATA)

  data = dict()
  data['train'] = dict()
  data['test'] = dict()

  for i in range(NUM_MODALITIES):
    data['train'][keys[i]] = X_train[:,i]@TFSTAR@TF_SEQ[keys[i]]
    data['test'][keys[i]] = X_test[:,i]@TFSTAR@TF_SEQ[keys[i]]
  data['train']['label'] = y_train
  data['test']['label'] = y_test


  with open(args.out_path, 'wb') as f:
    print(f"Saving to: {args.out_path}")
    pickle.dump(data, f)

else:

  total_raw_data = []

  for modality_idx in range(NUM_MODALITIES):
    temp_list = []
    for concept_idx_a in range(NUM_CONCEPTS):
      for concept_idx_b in range(NUM_CONCEPTS):
        sample_set_a = np.random.multivariate_normal(concept_means[concept_idx_a],np.eye(internal_concept_dim)*SEP, (NUM_DATA,))@TFSTAR@TF_SEQ[keys[modality_idx]]
        sample_set_b = np.random.multivariate_normal(concept_means[concept_idx_b],np.eye(internal_concept_dim)*SEP, (NUM_DATA,))@TFSTAR@TF_SEQ[keys[modality_idx]]
        sample_set = np.concatenate([sample_set_a, sample_set_b], axis=1)
      temp_list.append(sample_set)
    total_raw_data.append(temp_list)
  X_train = np.array(total_raw_data).transpose((2,1,0,3)).reshape((-1,NUM_MODALITIES, DIMENSION*2))
  y_train = np.tile([i for i in range(NUM_CONCEPTS*NUM_CONCEPTS)], NUM_DATA)

  total_test_data = []

  for modality_idx in range(NUM_MODALITIES):
    temp_list = []
    for concept_idx_a in range(NUM_CONCEPTS):
      for concept_idx_b in range(NUM_CONCEPTS):
        sample_set_a = np.random.multivariate_normal(concept_means[concept_idx_a],np.eye(internal_concept_dim)*SEP, (NUM_DATA,))@TFSTAR@TF_SEQ[keys[modality_idx]]
        sample_set_b = np.random.multivariate_normal(concept_means[concept_idx_b],np.eye(internal_concept_dim)*SEP, (NUM_DATA,))@TFSTAR@TF_SEQ[keys[modality_idx]]
        sample_set = np.concatenate([sample_set_a, sample_set_b], axis=1)
      temp_list.append(sample_set)
    total_test_data.append(temp_list)
  X_test = np.array(total_test_data).transpose((2,1,0,3)).reshape((-1,NUM_MODALITIES, DIMENSION*2))
  y_test = np.tile([i for i in range(NUM_CONCEPTS*NUM_CONCEPTS)], NUM_DATA)


  data = dict()
  data['train'] = dict()
  data['test'] = dict()

  for i in range(NUM_MODALITIES):
    data['train'][keys[i]] = X_train[:,i]
    data['test'][keys[i]] = X_test[:,i]
  data['train']['label'] = y_train
  data['test']['label'] = y_test


  with open(args.out_path, 'wb') as f:
    print(f"Saving to: {args.out_path}")
    pickle.dump(data, f)
