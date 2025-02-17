#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf


# Dataset for generating sequences, with labels predicting whether the cumulative sum
# is odd/even.
class Dataset:
    def __init__(self, sequences_num, sequence_length, sequence_dim, seed, shuffle_batches=True):
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.randint(0, max(2, sequence_dim), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32), "labels": labels}
        self._size = sequences_num

        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def batches(self, size=None):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
            batch_size = min(size or np.inf, len(permutation))
            batch_perm = permutation[:batch_size]
            permutation = permutation[batch_size:]

            batch = {}
            for key in self._data:
                batch[key] = self._data[key][batch_perm]
            yield batch


class Network:
    def __init__(self, args):
        # Construct the model.
        sequences = tf.keras.layers.Input(shape=[args.sequence_length, args.sequence_dim])
        # TODO: Process the sequence using a RNN with cell type `args.rnn_cell`
        # and with dimensionality `args.rnn_cell_dim`. Use `return_sequences=True`
        # to get outputs for all sequence elements.
        #
        # Prefer `tf.keras.layers.{LSTM,GRU,SimpleRNN}` to
        # `tf.keras.layers.RNN` wrapper with `tf.keras.layers.{LSTM,GRU,SimpleRNN}Cell`,
        # because the former can run transparently on a GPU and is also
        # considerably faster on a CPU).
        if args.rnn_cell == "SimpleRNN":
            layer1 = tf.keras.layers.SimpleRNN(args.rnn_cell_dim, return_sequences=True)(sequences)
        elif args.rnn_cell == "LSTM":
            layer1 = tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True)(sequences)
        elif args.rnn_cell == "GRU":
            layer1 = tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True)(sequences)
        else:
            print("Invalid layer Parameter")
            exit(1)

        # TODO: If `args.hidden_layer` is nonzero, process the result using
        # a ReLU-activated fully connected layer with `args.hidden_layer` units.

        if args.hidden_layer:
            layer1 = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(layer1)

        # TODO: Generate predictions using a fully connected layer
        # with one output and `tf.nn.sigmoid` activation.
        predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(layer1)

        self.model = tf.keras.Model(inputs=sequences, outputs=predictions)

        # TODO: Create an Adam optimizer in self._optimizer
        self._optimizer = tf.keras.optimizers.Adam()

        # TODO: Create a suitable loss in self._loss
        def loss_func(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            return tf.losses.MeanSquaredError(y_true, y_pred)

        self._loss = loss_func

        # TODO: Create two metrics in self._metrics dictionary:
        #  - "loss", which is tf.metrics.Mean()
        #  - "accuracy", which is suitable accuracy
        self._metrics = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.CosineSimilarity()}

        # TODO: Create a summary file writer using `tf.summary.create_file_writer`.
        # I usually add `flush_millis=10 * 1000` arguments to get the results reasonably quickly.
        logdir = os.path.join("tflogs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        ))
        self._writer = tf.summary.create_file_writer(logdir, flush_millis=10 * 1000)

    @tf.function
    def train_batch(self, batch, clip_gradient):
        # TODO: Using a gradient tape, compute
        # - probabilities from self.model, passing `training=True` to the model
        # - loss, using `self._loss`
        with tf.GradientTape() as tape:
            logits = self.model(batch["sequences"], training=True)
            loss_value = self._loss(y_true=batch["labels"], y_pred=logits)

        # TODO: Then, compute `gradients` using `tape.gradients` with the loss and model variables.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # TODO: If `args.clip_gradient` is nonzero, clip the gradient
        # and compute `gradient_norm` using `tf.clip_by_global_norm`.
        # Otherwise, only compute the `gradient_norm` using
        # `tf.linalg.global_norm`.
        if args.clip_gradient:
            gradient_norm = tf.clip_by_global_norm(grads, clip_gradient)
        else:
            gradient_norm = tf.linalg.global_norm(grads)

        # TODO: Apply the gradients using the `self._optimizer`
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # TODO: Generate the summaries. Start by setting the current summary step using
        # `tf.summary.experimental.set_step(self._optimizer.iterations)`.
        tf.summary.experimental.set_step(self._optimizer.iterations)

        # Then, in the following with block, which records summaries
        # each 100 steps, perform the following:
        with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 100 == 0):
            # - iterate through the self._metrics
            for key in self._metrics.keys():
                #   - reset each metric
                self._metrics[key].reset_states()

            #   - for "loss" metric, apply currently computed `loss`
            self._metrics["loss"].update_state(batch["labels"], logits)

            #   - for other metrics, compute their value using the gold labels and predictions
            self._metrics["accuracy"].update_state(batch["labels"], logits)

            #   - then, emit the summary to TensorBoard using
            #     `tf.summary.scalar("train/" + name, metric.result())`
            tf.summary.scalar("train/" + "accuracy", self._metrics["accuracy"].result())
            tf.summary.scalar("train/" + "loss", self._metrics["loss"].result())

            # - finally, also emit the gradient_norm using
            #   `tf.summary.scalar("train/gradient_norm", gradient_norm)`
            tf.summary.scalar("train/gradient_norm", gradient_norm)

            """if self._optimizer.iterations % 100 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (self._optimizer.iterations, float(loss_value))
                )"""

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch(batch, args.clip_gradient)

    @tf.function
    def predict_batch(self, batch):
        return self.model(batch["sequences"], training=False)

    def evaluate(self, dataset, args):
        # TODO: Similarly to training summaries, compute the metrics
        # averaged over all `dataset`.
        #
        # Start by resetting all metrics in `self._metrics`.
        for key in self._metrics.keys():
            self._metrics[key].reset_states()

        # TODO: Then iterate over all batches in `dataset.batches(args.batch_size)`.
        for batch in dataset.batches(args.batch_size):
            # - For each, predict probabilities using `self.predict_batch`.
            predicted = self.predict_batch(batch)

            # - Compute loss of the batch
            self._loss(predicted, batch["labels"])

            # - Update the metrics (the "loss" metric uses current loss, others are computed
            #   using the gold labels and the predictions)
            self._metrics["loss"].update_state(batch["labels"], predicted)
            self._metrics["accuracy"].update_state(batch["labels"], predicted)
        #
        # Finally, create a dictionary `metrics` with results, using names and values in `self._metrics`.
        metrics = {"loss": self._metrics["loss"].result(), "accuracy": self._metrics["accuracy"].result()}

        with self._writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar("test/" + name, metric)

        return metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--clip_gradient", default=0., type=float, help="Norm for gradient clipping.")
    parser.add_argument("--hidden_layer", default=0, type=int, help="Additional hidden layer after RNN.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
    parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--test_sequences", default=1000, type=int, help="Number of testing sequences.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42, shuffle_batches=True)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43, shuffle_batches=False)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(train, args)
        metrics = network.evaluate(test, args)
    with open("sequence_classification.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
