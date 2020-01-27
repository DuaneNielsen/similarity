import torch
from datasets import package
from torch.utils.data.dataloader import DataLoader
from models import autoencoder, maker, similarity
from config import config

if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ data """
    datapack = package.datasets[args.dataset_name]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train_l = DataLoader(train, batch_size=4, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder = maker.make_layers(args.model_encoder, type=args.model_type)
    decoder = maker.make_layers(args.model_decoder, type=args.model_type)
    auto_encoder = autoencoder.LinearAutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)
    auto_encoder.load_state_dict(torch.load(args.load))

    for image, label in train_l:
        image = image.to(args.device)
        print(similarity.similarity_trace_mean(auto_encoder.encoder,
                                               image[0:2].flatten(start_dim=1),
                                               image[2:4].flatten(start_dim=1)))