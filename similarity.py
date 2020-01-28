import torch
from datasets import package
from torch.utils.data.dataloader import DataLoader
from models import autoencoder, maker, similarity
from config import config
from tqdm import tqdm

if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ data """
    datapack = package.datasets[args.dataset_name]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train_l = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder = maker.make_layers(args.model_encoder, type=args.model_type)
    decoder = maker.make_layers(args.model_decoder, type=args.model_type)
    auto_encoder = autoencoder.LinearAutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)
    auto_encoder.load_state_dict(torch.load(args.load))

    batch = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for images, labels in tqdm(train_l):
        for image, label in zip(images, labels):
            batch[label.item()].append(image)

    sim_stats = torch.zeros(len(batch), len(batch))

    for numeral_left in tqdm(batch):
        for numeral_right in batch:
            length = min(len(batch[numeral_left]), len(batch[numeral_right]), 50)
            left = torch.stack(batch[numeral_left][0:length]).to(args.device)
            right = torch.stack(batch[numeral_right][0:length]).to(args.device)
            sim_stats[numeral_left, numeral_right] = similarity.similarity_trace_mean(auto_encoder.encoder,
                                                                                      left.flatten(start_dim=1),
                                                                                      right.flatten(start_dim=1)).mean()
    torch.set_printoptions(sci_mode=False)
    print(sim_stats)
