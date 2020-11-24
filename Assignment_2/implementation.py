import tensorflow as tf
import random
import re

# 77% 128 400 50


BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 400  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review = review.replace("can't", "can not")
    review = review.replace("haven't", "have not")
    processed_review = review.lower()
    reObj1 = re.compile('(\w+)')
    processed_review = reObj1.findall(processed_review)

    processed_review = [i for i in processed_review if i not in stop_words and len(i) > 2]
    random.shuffle(processed_review)

    if len(processed_review) < MAX_WORDS_IN_REVIEW:
        for i in range(len(processed_review), MAX_WORDS_IN_REVIEW):
            processed_review.append("during")


    return ' '.join(processed_review)



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    #
    #input
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name = "input_data")
    #labels
    labels = tf.placeholder(tf.float32, [None, 2], name = "labels")
    #drop_out
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

    weights = tf.Variable(tf.truncated_normal([100, 2], stddev = 0.1))
    biases = tf.Variable(tf.constant(0.1, shape = [2]))

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(100)

    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = dropout_keep_prob)

    init = drop_cell.zero_state(BATCH_SIZE, tf.float32)
    #cells = []
    #for _ in range(3):
    #    cells.append(tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = dropout_keep_prob))
    #drop_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    outputs, final_state = tf.nn.dynamic_rnn(drop_cell, input_data, dtype = tf.float32)
    results = tf.nn.sigmoid(tf.matmul(final_state[1], weights) + biases)
    #loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = results, labels = labels), name = "loss")
    #optimizer
    #1
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(loss)
    #optimizer = tf.train.AdagradDAOptimizer(0.001).minimize(loss)
    #
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #very low
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    #0.90-0.95

    #accuracy
    correct_prediction = tf.equal(tf.argmax(results, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
