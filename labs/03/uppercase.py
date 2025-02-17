#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

if __name__ == "__main__":
    # Parse arguments
    # TODO: Set reasonable values for `alphabet_size` and `window`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=100, type=int,
                        help="If nonzero, limit alphabet to this many most frequent chars.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--window", default=5, type=int, help="Window size to use.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is representedy by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.

    input = tf.keras.layers.Input(input_shape=[2 * args.window + 1], dtype=tf.int32)
    one_hot = tf.keras.layers.Embeddign(len(uppercase_data.Dataset.alphabet),
                                        len(uppercase_data.Dataset.alphabet))(input)
    dense_layer = tf.keras.layers.Dense(10, activation='relu', regularization=tf.keras.regularizers.l1())(one_hot)
    output = tf.keras.layers.Dense(2, activation='sigmoid')(dense_layer)

    model = tf.keras.Model(input=input, output=output)
    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.BinaryCrossentropy()
                  , metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(x=train_data, y=label, batch_size=args.batch_size, epochs=args.epoch)

    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Generate correctly capitalized test set.
        # Use `uppercase_data.test.text` as input, capitalize suitable characters,
        # and write the result to `uppercase_test.txt` file.
        pass
