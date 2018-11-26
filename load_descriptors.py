import numpy
import os


def load_file(filename, num_vectors, vector_dimensions):
    assert os.path.isfile(filename), "no existe archivo " + filename
    mat = numpy.fromfile(filename, dtype=numpy.float32)
    return numpy.reshape(mat, (num_vectors, vector_dimensions))


def load_dataset_pair(dirname, num_vectors_q, num_vectors_r, vector_dimensions):
    file_q = "{}/Q-{}_{}_4F.bin".format(dirname, num_vectors_q, vector_dimensions)
    file_r = "{}/R-{}_{}_4F.bin".format(dirname, num_vectors_r, vector_dimensions)
    data_q = load_file(file_q, num_vectors_q, vector_dimensions)
    data_r = load_file(file_r, num_vectors_r, vector_dimensions)
    return data_q, data_r


(dataset_q, dataset_r) = load_dataset_pair("descriptores/MEL128", 21573, 33545, 128)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

(dataset_q, dataset_r) = load_dataset_pair("descriptores/SIFT", 2886, 202088, 128)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))

(dataset_q, dataset_r) = load_dataset_pair("descriptores/VGG19", 842, 10171, 4096)
print("Q={} R={}".format(dataset_q.shape, dataset_r.shape))
