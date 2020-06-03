#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # TODO: Set reasonable defaults and possibly add more arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
    parser.add_argument("--model", default="gym_cartpole_model.h5", type=str, help="Output model path.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--hidden_layer", default=100, type=float, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.16, type=float, help="Initial learning rate.")
    parser.add_argument("--dropout", default=0.567, type=float, help="Dropout rate.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

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

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[:-1]])
            labels.append(int(columns[-1]))

    observations, labels = np.array(observations), np.array(labels)
    labels = np_utils.to_categorical(labels)
    # TODO: Create the model in the `model` variable. Note that
    # the model can perform any of:
    # - binary classification with 1 output and sigmoid activation;
    # - two-class classification with 2 outputs and softmax activation.

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2()),
         tf.keras.layers.Dropout(args.dropout),
         tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2()),
         tf.keras.layers.Dropout(args.dropout),
         #tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2()),
         #tf.keras.layers.Dropout(args.dropout),
         tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
         ]
    )

    # TODO: Prepare the model for training using the `model.compile` method.
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=100 / args.batch_size,
        end_learning_rate=1e-7,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.6)
    model.compile(optimizer=optimizer,
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=[tf.metrics.BinaryAccuracy()],
                  )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
    model.fit(
        observations, labels,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_split=0.2,
        callbacks=[tb_callback]
    )

    # Save the model, without the optimizer state.
    model.save(args.model, include_optimizer=False)
