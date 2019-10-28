import sys
import numpy
import tensorflow as tf


USC_EMAIL = 'ramisett@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = '*********************'  # TODO(student): You will be given a password via email.


class DatasetReader(object):

    # TODO(student): You must implement this.
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.

        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!

        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...]
        """

        parsed_file_dict = []
        file = open(filename)
        lines = file.readlines()
#        print(lines)
        for line in lines:
            line_list = []
            for word in line.split():
                term = word[ : word.rindex("/")]
                tag = word[word.rindex("/")+1 : ]
                temp1 = None
                temp2 = None
                if term not in term_index:
                    temp1 = len(term_index)
                    term_index[term] = len(term_index)
                else:
                    temp1 = term_index[term]

                if tag not in tag_index:
                    temp2 = len(tag_index)
                    tag_index[tag] = len(tag_index)
                else:
                    temp2 = tag_index[tag]

                line_list.append((temp1, temp2))
            parsed_file_dict.append(line_list)
        return parsed_file_dict
        # pass

    # TODO(student): You must implement this.
    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]],

                [2, 4]
            )
        """
        terms_matrix = []
        tags_matrix = []
        lengths = numpy.array([len(x) for x in dataset])
        max_length = max(lengths)
        for i in range(len(dataset)):
            temp1 = []
            temp2 = []
            for j in range(max_length):
                if j < lengths[i]:
                    temp1.append(dataset[i][j][0])
                    temp2.append(dataset[i][j][1])
                else:
                    temp1.append(0)
                    temp2.append(0)
            terms_matrix.append(temp1)
            tags_matrix.append(temp2)

        return numpy.array(terms_matrix), numpy.array(tags_matrix), numpy.array(lengths)
        # pass

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).

        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                            (train_terms, train_tags, train_lengths),
                            (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        You will be passed these arguments by the grader script.
        """
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int64, [None], 'lengths')

    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        return tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        pass

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        pass

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).

        Please do not change or override self.x nor self.lengths in this function.

        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """

        state_size = 50
        embeddings = tf.get_variable('embeddings', [self.num_terms, 30], trainable=True)
        x_embeddings = tf.nn.embedding_lookup(embeddings, self.x)
        # rnn_cell = tf.keras.layers.SimpleRNNCell(state_size)
        # states = []
        # cur_state = tf.zeros(shape=[1, state_size])
        # for i in range(self.max_length):
        #      cur_state = rnn_cell(x_embeddings[:, i, :], [cur_state])[0] # shape (batch, state_size)
        #      states.append(cur_state)
        # stacked_states = tf.stack(states,axis=1)
        # self.logits = tf.contrib.layers.fully_connected(stacked_states, self.num_tags, trainable=True)
                
        fw_cell = tf.keras.layers.LSTMCell(state_size)
        bw_cell = tf.keras.layers.LSTMCell(state_size)
        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x_embeddings, dtype=tf.float32,scope='LSTM')
        concat_op = tf.concat([forward_output, backward_output], axis=2)
        self.logits = tf.contrib.layers.fully_connected(concat_op, self.num_tags, trainable=True)
        
        # TODO(student): make logits an RNN on x.
        # self.logits = tf.zeros([tf.shape(self.x)[0], self.max_length, self.num_tags])

    # TODO(student): You must implement this.
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.

        Hint: This function is straight forward and you might find this code useful:

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """

        logits = self.session.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

#        return numpy.zeros_like(terms)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.

        It is up to you how you implement this function, as long as train_on_batch
        works.

        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """
        self.learning_rate = tf.placeholder_with_default(numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
        self.predicted_tags = tf.placeholder(tf.int64, [None, self.max_length], 'predicted_tags')
        weights = self.lengths_vector_to_binary_matrix(self.lengths)
#        l = tf.contrib.seq2seq.sequence_loss(self.logits, self.predicted_tags, weights)
        
        
        step_grad = tf.Variable(0, trainable=False)
        learn_rate = tf.train.exponential_decay(0.01, step_grad, 1000, 0.9, staircase=False)
        loss_val = tf.contrib.seq2seq.sequence_loss(self.logits, self.predicted_tags, weights)
        loss = tf.reduce_mean(loss_val)
        
        

#        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
#        self.train_op = tf.contrib.training.create_train_op(l, tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate))

        self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
#        pass

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=1e-7):
        """Performs updates on the model given training training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        # <-- Your implementation goes here.
        # Finally, make sure you uncomment the `return True` below.
        # return True
        def batch_step(batch_x, batch_y, lengths, lr):
            self.session.run(self.train_op, {
                    self.x: batch_x,
                    self.lengths: lengths,
                    self.predicted_tags: batch_y,
                    self.learning_rate: lr,
            })
    
        def step(lr=0.01, batch_size=32):
            indices = numpy.random.permutation(terms.shape[0])
            for si in range(0, terms.shape[0], batch_size):
                se = min(si + batch_size, terms.shape[0])
                slice_x = terms[indices[si:se]] + 0    # + 0 to copy slice
                batch_step(slice_x, tags[indices[si:se]], lengths[indices[si:se]], lr)
    
    
        lr = 0.05
        step(lr)
#        print('Training model ... ')
#        for j in range(300): step(lr)
#        for j in range(300): step(lr/2)
#        for j in range(300): step(lr/4)
#        print('Results from training:')
        return True
#        pass

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j+1))
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()
