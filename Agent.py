import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper

distr = tf.contrib.distributions


class Agent(object):

    def __init__(self, config, input_, type = 0, label = None):

        self.config = config
        if(type==1):
            self.input_ = input_
        else:
            self.input_ = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.max_length, self.config.input_dimension))
        self.type = type
        if type != 1:
            self.sp_train(label)
    def compute(self):

        with tf.variable_scope("encoder"):
            self.Encoder = Attentive_encoder(self.config)
            self.encoder_output = self.Encoder.encode(self.input_)
        with tf.variable_scope('decoder'):
            # Ptr-net returns permutations (self.positions), with their log-probability for backprop
            self.ptr = Pointer_decoder(self.encoder_output, self.config)
        self.positions, self.log_softmax , self.pointing, self.mask_score, self.scores= self.ptr.loop_decode()
        return self.positions, self.log_softmax, self.pointing, self.mask_score, self.scores

    def sp_train(self, label):

        self.compute()
        self.label = label
        self.loss = 0


        for i in range(label.shape[0]):
            self.loss += tf.nn.softmax_cross_entropy_with_logits(logits=self.mask_score[i], labels=self.label[i])

        self.soft_max = tf.nn.softmax(self.mask_score)

        self.loss_by_me = -label*tf.log(self.soft_max+1e-5)

        self.opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        self.train_op = self.opt.minimize(self.loss_by_me)

        self.reduce_loss = tf.reduce_sum(self.loss_by_me)/(self.config.batch_size * self.config.max_length)

def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.variable_scope("multihead_attention", reuse=None):
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # num_heads*[batch_size, seq_length, seq_length]

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]

        # Residual connection
        outputs += inputs  # [batch_size, seq_length, n_hidden]

        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln',
                                                reuse=None)  # [batch_size, seq_length, n_hidden]

    return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs

def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu,
                  "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln',
                                                reuse=None)  # [batch_size, seq_length, n_hidden]

    return outputs


class Attentive_encoder(object):

    def __init__(self, config):
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of a city (coordinates)

        self.input_embed = config.hidden_dim  # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks

        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer
        self.is_training = not config.inference_mode

    #  with tf.name_scope('encode_'):
    # self.encode()

    def encode(self, inputs):
        # Tensor blocks holding the input sequences [Batch Size, Sequence Length, Features]
        # self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_raw")
        with tf.variable_scope("embedding"):
            # Embed input sequence
            W_embed = tf.get_variable("weights", [1, self.input_dimension, self.input_embed],
                                      initializer=self.initializer)
            self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            # Batch Normalization
            self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training,
                                                     name='layer_norm', reuse=None)

        with tf.variable_scope("stack"):
            # Blocks
            for i in range(self.num_stacks):  # num blocks
                with tf.variable_scope("block_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads,
                                                   dropout_rate=0.1, is_training=self.is_training)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * self.input_embed, self.input_embed],
                                           is_training=self.is_training)

            # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
            self.encoder_output = self.enc  ### NOTE: encoder_output is the ref for attention ###
            return self.encoder_output




# RNN decoder for pointer network
class Pointer_decoder(object):

    def __init__(self, encoder_output, config):
        #######################################
        ########## Reference vectors ##########
        #######################################

        self.encoder_output = encoder_output  # Tensor [Batch size x time steps x cell.state_size] to attend to
        self.h = tf.transpose(self.encoder_output, [1, 0,
                                                    2])  # [Batch size x time steps x cell.state_size] to [time steps x Batch size x cell.state_size]

        ############################
        ########## Config ##########
        ############################

        batch_size = encoder_output.get_shape().as_list()[0]  # batch size
        self.seq_length = encoder_output.get_shape().as_list()[1]  # sequence length
        n_hidden = encoder_output.get_shape().as_list()[2]  # num_neurons

        self.inference_mode = config.inference_mode  # True for inference, False for training
        self.temperature = config.temperature  # temperature parameter
        self.C = config.C  # logit clip

        ##########################################
        ########## Decoder's parameters ##########
        ##########################################

        # Variables initializer
        initializer = tf.contrib.layers.xavier_initializer()

        # Decoder LSTM cell
        self.cell = LSTMCell(n_hidden, initializer=initializer)

        # Decoder initial input is 'GO', a variable tensor
        first_input = tf.get_variable("GO", [1, n_hidden], initializer=initializer)
        self.decoder_first_input = tf.tile(first_input, [batch_size, 1])

        # Decoder initial state (tuple) is trainable
        first_state = tf.get_variable("GO_state1", [1, n_hidden], initializer=initializer)
        self.decoder_initial_state = tf.tile(first_state, [batch_size, 1]), tf.reduce_mean(self.encoder_output, 1)

        # Attending mechanism
        with tf.variable_scope("glimpse") as glimpse:
            self.W_ref_g = tf.get_variable("W_ref_g", [1, n_hidden, n_hidden], initializer=initializer)
            self.W_q_g = tf.get_variable("W_q_g", [n_hidden, n_hidden], initializer=initializer)
            self.v_g = tf.get_variable("v_g", [n_hidden], initializer=initializer)

        # Pointing mechanism
        with tf.variable_scope("pointer") as pointer:
            self.W_ref = tf.get_variable("W_ref", [1, n_hidden, n_hidden], initializer=initializer)
            self.W_q = tf.get_variable("W_q", [n_hidden, n_hidden], initializer=initializer)
            self.v = tf.get_variable("v", [n_hidden], initializer=initializer)

        ######################################
        ########## Decoder's output ##########
        ######################################

        self.log_softmax = []  # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = []  # store visited cities for reward
        self.attending = []  # for vizualition
        self.pointing = []  # for vizualition
        self.mask_score = []
        self.scores = []

        ########################################
        ########## Initialize process ##########
        ########################################

        # Keep track of first city
        self.first_city_hot = 0  ###########

        # Keep track of visited cities
        self.mask = 0

    # From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    # predict a distribution over next decoder input
    def attention(self, ref, query):

        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, self.W_ref_g, 1, "VALID",
                                     name="encoded_ref_g")  # [Batch size, seq_length, n_hidden]
        encoded_query_g = tf.expand_dims(tf.matmul(query, self.W_q_g, name="encoded_query_g"),
                                         1)  # [Batch size, 1, n_hidden]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1],
                                 name="scores_g")  # [Batch size, seq_length]

        # Attend to current city and cities to visit only (Apply mask)
        attention_g = tf.nn.softmax(scores_g - 1000. * (self.mask - self.first_city_hot),
                                    name="attention_g")  ###########
        self.attending.append(attention_g)

        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g, 2))
        glimpse = tf.reduce_sum(glimpse, 1) + query  ########### Residual connection

        # Pointing mechanism with 1 glimpse
        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID",
                                   name="encoded_ref")  # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(glimpse, self.W_q, name="encoded_query"),
                                       1)  # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1],
                               name="scores")  # [Batch size, seq_length]
        if self.inference_mode == True:
            scores = scores / self.temperature  # control diversity of sampling (inference mode)
        scores = self.C * tf.tanh(scores)  # control entropy

        # Point to cities to visit only (Apply mask)
        masked_scores = scores - 1000. * self.mask  # [Batch size, seq_length]
        pointing = tf.nn.softmax(masked_scores, name="attention")  # [Batch size, Seq_length]
        self.pointing.append(pointing)
        self.mask_score.append(masked_scores)
        self.scores.append(scores)

        return masked_scores

    # One pass of the decode mechanism
    def decode(self, prev_state, prev_input, timestep):
        with tf.variable_scope("loop"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()

            # Run the cell on a combination of the previous input and state
            output, state = self.cell(prev_input, prev_state)

            # Attention mechanism
            masked_scores = self.attention(self.encoder_output, output)

            # Multinomial distribution
            prob = distr.Categorical(masked_scores)

            # Sample from distribution
            position = prob.sample()
            position = tf.cast(position, tf.int32)
            if timestep == 0:
                self.first_city = position
                self.first_city_hot = tf.one_hot(self.first_city, self.seq_length)  ###########
            self.positions.append(position)

            # Store log_prob for backprop
            self.log_softmax.append(prob.log_prob(position))

            # Update mask
            self.mask = self.mask + tf.one_hot(position, self.seq_length)

            # Retrieve decoder's new input
            new_decoder_input = tf.gather(self.h, position)[0]

            return state, new_decoder_input

    def loop_decode(self):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Loop the decoding process and collect results
        s, i = self.decoder_initial_state, tf.cast(self.decoder_first_input, tf.float32)
        for step in range(self.seq_length):
            s, i = self.decode(s, i, step)

        # Return to start
        self.positions.append(self.first_city)

        # Stack visited indices
        self.positions = tf.stack(self.positions, axis=1)  # [Batch,seq_length+1]

        # Sum log_softmax over output steps
        self.log_softmax = tf.add_n(self.log_softmax)  # [Batch,seq_length]

        # Stack attending & pointing distribution
        self.attending = tf.stack(self.attending, axis=1)  # [Batch,seq_length,seq_length]
        self.pointing = tf.stack(self.pointing, axis=1)  # [Batch,seq_length,seq_length]
        self.mask_score = tf.stack(self.mask_score, axis=1)
        self.scores = tf.stack(self.scores, axis=1)
        # Return stacked lists of visited_indices and log_softmax for backprop
        return self.positions, self.log_softmax, self.pointing, self.mask_score, self.scores


