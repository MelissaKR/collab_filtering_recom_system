from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.factorization import WALSMatrixFactorization

tf.logging.set_verbosity(tf.logging.INFO)

def read_dataset(mode, args):
    def decode_example(protos, vocab_size):
        features = {
            "key": tf.FixedLenFeature(shape = [1], dtype = tf.int64),
            "indices": tf.VarLenFeature(dtype = tf.int64),
            "values": tf.VarLenFeature(dtype = tf.float32)}
        parsed_features = tf.parse_single_example(serialized = protos, features = features)
        values = tf.sparse_merge(sp_ids = parsed_features["indices"],
                                 sp_values = parsed_features["values"],
                                 vocab_size = vocab_size)

        # Save key to remap after batching
        key = parsed_features["key"]
        decoded_sparse_tensor = tf.SparseTensor(indices = tf.concat(values = [values.indices, [key]], axis = 0),
                                                values = tf.concat(values = [values.values, [0.0]], axis = 0),
                                                dense_shape = values.dense_shape)
        return decoded_sparse_tensor

    def remap_keys(sparse_tensor):
        # Current indices of our SparseTensor that we need to fix
        bad_indices = sparse_tensor.indices

        # Current values of our SparseTensor that we need to fix
        bad_values = sparse_tensor.values

        store_mask = tf.concat(values = [bad_indices[1:,0] - bad_indices[:-1,0],
                                         tf.constant(value = [1], dtype = tf.int64)], axis = 0)

        ### Mask out the store rows from the values
        good_values = tf.boolean_mask(tensor = bad_values, mask = tf.equal(x = store_mask, y = 0))
        item_indices = tf.boolean_mask(tensor = bad_indices, mask = tf.equal(x = store_mask, y = 0))
        store_indices = tf.boolean_mask(tensor = bad_indices, mask = tf.equal(x = store_mask, y = 1))[:, 1]
        good_store_indices = tf.gather(params = store_indices, indices = item_indices[:,0])

        ### store and item indices are rank 1, need to make rank 1 to concat
        good_store_indices_expanded = tf.expand_dims(input = good_store_indices, axis = -1)
        good_item_indices_expanded = tf.expand_dims(input = item_indices[:, 1], axis = -1)
        good_indices = tf.concat(values = [good_store_indices_expanded, good_item_indices_expanded], axis = 1)

        remapped_sparse_tensor = tf.SparseTensor(indices = good_indices, values = good_values, dense_shape = sparse_tensor.dense_shape)
        return remapped_sparse_tensor

    def parse_tfrecords(filename, vocab_size):
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
        else:
            num_epochs = 1

        files = tf.gfile.Glob(filename = os.path.join(args["input_path"], filename))

        # Create dataset from file list
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(map_func = lambda x: decode_example(x, vocab_size))
        dataset = dataset.repeat(count = num_epochs)
        dataset = dataset.batch(batch_size = args["batch_size"])
        dataset = dataset.map(map_func = lambda x: remap_keys(x))
        return dataset.make_one_shot_iterator().get_next()

    def _input_fn():
        features = {
            WALSMatrixFactorization.INPUT_ROWS: parse_tfrecords("items_for_store", args["nitems"]),
            WALSMatrixFactorization.INPUT_COLS: parse_tfrecords("stores_for_item", args["nstores"]),
            WALSMatrixFactorization.PROJECT_ROW: tf.constant(True)
        }
        return features, None

    return _input_fn



def find_top_k_ind(store, item_factors, k):
    all_items = tf.matmul(a = tf.expand_dims(input = store, axis = 0),
                          b = tf.transpose(a = item_factors))
    topk = tf.nn.top_k(input = all_items, k = k)

    return tf.cast(x = topk.indices, dtype = tf.int64)


def find_top_k_val(store, item_factors, k):
    all_items = tf.matmul(a = tf.expand_dims(input = store, axis = 0),
                          b = tf.transpose(a = item_factors))
    topk = tf.nn.top_k(input = all_items, k = k)

    return tf.cast(x = topk.values, dtype = tf.float32)


def batch_predict(args):
    import numpy as np
    with tf.Session() as sess:
        estimator = tf.contrib.factorization.WALSMatrixFactorization(
            num_rows = args["nstores"],
            num_cols = args["nitems"],
            embedding_dimension = args["n_embeds"],
            model_dir = args["output_dir"])


        store_factors = tf.convert_to_tensor(value = estimator.get_row_factors()[0]) # (nstores, nembeds)
        item_factors = tf.convert_to_tensor(value = estimator.get_col_factors()[0])# (nitems, nembeds)


        # For each store, find the top K items
        topk = tf.squeeze(input = tf.map_fn(fn = lambda store: find_top_k_ind(store, item_factors ,args["topk"]),
                                            elems = store_factors,
                                            dtype = tf.int64))
        topk_val = tf.squeeze(input = tf.map_fn(fn = lambda store: find_top_k_val(store, item_factors ,args["topk"]),
                                            elems = store_factors,
                                            dtype = tf.float32))
        with file_io.FileIO(os.path.join(args["output_dir"], "batch_pred.txt"), mode = 'w') as f:
            for best_items_for_store in list(zip(topk.eval(), topk_val.eval())):
                f.write(",".join(str(x) for x in best_items_for_store) + '\n')


def train_and_evaluate(args):
    train_steps = int(0.5 + (1.0 * args["num_epochs"] * args["nstores"]) / args["batch_size"])
    steps_in_epoch = int(0.5 + args["nstores"] / args["batch_size"])
    print("Will train for {} steps, evaluating once every {} steps".format(train_steps, steps_in_epoch))

    def experiment_fn(output_dir):
        return tf.contrib.learn.Experiment(
            tf.contrib.factorization.WALSMatrixFactorization(
                num_rows = args["nstores"],
                num_cols = args["nitems"],
                embedding_dimension = args["n_embeds"],
                regularization_coeff = 0.2,
                model_dir = args["output_dir"]),
            train_input_fn = read_dataset(tf.estimator.ModeKeys.TRAIN, args),
            eval_input_fn = read_dataset(tf.estimator.ModeKeys.EVAL, args),
            train_steps = train_steps,
            eval_steps = 1,
            min_eval_frequency = steps_in_epoch
        )

    from tensorflow.contrib.learn.python.learn import learn_runner
    learn_runner.run(experiment_fn = experiment_fn, output_dir = args["output_dir"])

    batch_predict(args)
