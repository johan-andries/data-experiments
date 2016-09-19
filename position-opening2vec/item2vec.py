import collections
import itertools
import math
import datetime

import numpy as np
import tensorflow as tf


def create_batch_generator(list_of_item_idxs_in_session, batch_size):
    def next():
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      for i in range(batch_size):
        current_item_set = list_of_item_idxs_in_session[next.session_index]
        index1, index2 = np.random.choice(len(current_item_set), 2, replace=False)
        batch[i] = current_item_set[index1]
        labels[i,0] = current_item_set[index2]
        next.session_index = (next.session_index+1) % len(list_of_item_idxs_in_session)
      return batch, labels

    next.session_index = 0
    return next

def run(num_steps, next_batch, lr, batch_size, num_sampled, vocabulary_size, embedding_size):
    #lr = 0.5
    #embedding_size = 50  # Dimension of the embedding vector.
    #batch_size = 128
    #num_sampled = 128   # Number of negative examples to sample.
    #vocabulary_size = len(unique_relevant_vac_ids)

    graph = tf.Graph()

    with graph.as_default():

      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        init_width = 0.5 / embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -init_width, init_width))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


      loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))

      optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm

      # Add variable initializer.
      init = tf.initialize_all_variables()


    # Step 5: Begin training.
    #num_steps = 2000001

    with tf.Session(graph=graph) as session:

      init.run()

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_labels = next_batch()
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 20000 == 0:
          if step > 0:
            average_loss /= 20000
          print "%s Average loss at step %s: %s\n" % (datetime.datetime.now(), step, average_loss)
          average_loss = 0

      return normalized_embeddings.eval()
