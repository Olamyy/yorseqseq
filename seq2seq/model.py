from keras import Input, Model
from keras.layers import LSTM, Dense

from constants import num_samples, latent_dimensions, epochs, batch_size
from seq2seq.base import BaseSeq2Seq
from utils import read_corpus


class FCholletSeq2Seq(BaseSeq2Seq):
    def __init__(self, data_path, fformat="dir", nlevel='char', nump_samples=num_samples, latent_dimension=latent_dimensions, epoch=epochs, batch=batch_size):
        super().__init__(data_path, fformat, nlevel, nump_samples, latent_dimension, epoch, batch)

    def setup(self):
        self.input_characters, self.target_characters, self.input_text, self.target_text = read_corpus(self.data_path, self.fformat, self.nlevel)

        print(self.input_characters)


        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_text])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_text])

    def build_model(self):
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dimension, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(self.latent_dimension, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model(inputs=[encoder_inputs, decoder_inputs],
                      outputs=decoder_outputs)
        return model


class CustomSeq2Seq(FCholletSeq2Seq):
    pass

