import numpy

from constants import num_samples, latent_dimensions, epochs, batch_size


class BaseSeq2Seq(object):
    def __init__(self, data_path, fformat="dir", nlevel='char', prep=False, num_samples=num_samples, latent_dimension=latent_dimensions, epoch=epochs, batch=batch_size):
        self.data_path = data_path
        self.fformat = fformat
        self.nlevel = nlevel
        self.batch = batch
        self.latent_dimension = latent_dimension
        self.epochs = epochs
        self.model = None
        self.history = None
        self.input_text = []
        self.target_text = []
        self.input_characters = ()
        self.target_characters = ()
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None
        self.num_samples = num_samples
        self.prep = prep
        self.setup()
        self.params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'num_samples': num_samples,
            'max_encoder_seq_length': self.max_encoder_seq_length,
            'max_decoder_seq_length': self.max_decoder_seq_length,
            'num_encoder_tokens': self.num_encoder_tokens,
            'num_decoder_tokens': self.num_decoder_tokens,
            'latent_dimensions': self.latent_dimension,
            'nlevel': self.nlevel,
        }

    def __str__(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        param_string = ", ".join(
            '%s=%s' % (k, v)
            for k, v in self.get_params().items()
            if k in ['nlevel', 'num_samples', 'max_encoder_seq_length', 'max_decoder_seq_length', 'num_encoder_tokens', 'num_decoder_tokens', 'latent_dimensions']
        )
        return "%s(%s)" % (class_name, param_string)

    def setup(self):
        """
        Override this.
        :return:
        """
        pass

    def build_model(self):
        """
        Override this.
        :return:
        """
        pass

    def tokenize(self):
        """

        :return:
        """
        input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])
        return input_token_index, target_token_index

    def to_numeric_data(self):
        """

        :return:
        """
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        input_token_index, target_token_index = self.tokenize()
        encoder_input_data = numpy.zeros(
            (len(self.input_text), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = numpy.zeros(
            (len(self.input_text), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = numpy.zeros(
            (len(self.input_text), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_text, self.target_text)):
            for t, char in enumerate(self.input_text):
                pass
            #     encoder_input_data[i, t, input_token_index[char]] = 1.
            # for t, char in enumerate(target_text):
            #     decoder_input_data[i, t, target_token_index[char]] = 1.
            #     if t > 0:
            #         decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

    def fit(self, save_to=None):
        """

        :param save_to:
        :return:
        """
        self.validate_params()
        model = self.build_model()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        encoder_input_data, decoder_input_data, decoder_target_data = self.to_numeric_data()
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        model.save('{}.h5'.format(save_to if save_to else 'model'))

    def infer(self):
        """

        :return:
        """
        pass

    def validate_params(self):
        """

        :return:
        """
        checks = []  # Run some checks here. Make sure everything you need to compile the model is available.
        if any([i is None for i in checks]):
            raise ValueError(
                "Setup not complete. Must set all of {}".format(str([i for i in checks])))

    def get_params(self, deep=None):
        """

        :param deep:
        :return:
        """
        return self.params
