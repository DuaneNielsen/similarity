from models.mnn import Container
import torch

class AutoEncoder(Container):
    def __init__(self, encoder, decoder,
                 init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x

    def load(self, root_path):
        self.encoder.load(root_path + '/encoder')
        self.decoder.load(root_path + '/decoder')

    def save(self, root_path):
        self.encoder.save(root_path + '/encoder')
        self.decoder.save(root_path + '/decoder')


class LinearAutoEncoder(Container):
    def __init__(self, encoder, decoder,
                 init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        z = self.encoder(torch.flatten(x, start_dim=1))
        x_ = self.decoder(z)
        return z, x_.reshape(x.shape)

    def load(self, root_path):
        self.encoder.load(root_path + '/encoder')
        self.decoder.load(root_path + '/decoder')

    def save(self, root_path):
        self.encoder.save(root_path + '/encoder')
        self.decoder.save(root_path + '/decoder')


