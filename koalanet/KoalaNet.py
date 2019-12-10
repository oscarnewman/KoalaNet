# import statements
import argparse
import os
import time

import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from dataloader import RawImageDataset
from networks import KoalaNet

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a fully convolutional network to brighten under-exposed raw images"
    )

    parser.add_argument("--testwith", type=str, dest='testwith', default="data/test",
                        help="The dir for the test data")
    parser.add_argument("--trainwith", type=str, dest='trainwith', default="data/test",
                        help="The dir for the train data")

    parser.add_argument('-A', '--lr', type=float, default=0.001, help="The learning rate")
    parser.add_argument('-E', '--epochs', type=int, default=5, help="num epochs")
    parser.add_argument('-B', '--batch', type=int, default=2, help='batch size')
    parser.add_argument('--resume', type=str, default=None, help='batch size')

    args = parser.parse_args()

    # add code for datasets (we always use train and validation/ test set)
    data_transforms = transforms.Compose([
        # transforms.ToPILImage()

        # transforms.Resize((opt.img_size, opt.img_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = RawImageDataset(manifest_csv=os.path.join(args.trainwith, 'manifest.csv'),
                                    root_dir=args.trainwith,
                                    crop=512
                                    )
    train_data_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=7)
    train_data_preloader = data.DataLoader(train_dataset, batch_size=1, num_workers=0)

    test_dataset = RawImageDataset(manifest_csv=os.path.join(args.testwith, 'manifest.csv'),
                                   root_dir=args.testwith,
                                   crop=1024
                                   )
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # instantiate network (which has been imported from *networks.py*)
    net = KoalaNet()

    # create losses (criterion in pytorch)
    criterion_L1 = torch.nn.L1Loss()
    criterion_L2 = torch.nn.MSELoss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        print("Cuda is avaiable and being used\n")

    # create optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume)  # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()

    # Load our data first
    print("Preloading data for efficient epochs")
    print("====================================")
    pbar = tqdm(enumerate(BackgroundGenerator(train_data_preloader, max_prefetch=8)),
                total=len(train_data_preloader))
    for i, data in pbar:
        pass

    cetotal = 0
    cenum = 0

    print(f"\nBeginning Training for {args.epochs} epochs")
    print("===============================================")
    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, args.epochs):
        # set models to train mode
        net.train()
        num_iters = len(train_data_loader)

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_data_loader),
                    total=len(train_data_loader))
        start_time = time.time()

        avg_loss = 0

        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            light_img = data['light']
            dark_img = data['dark']
            # dark_rgb = data['dark_rgb']
            if use_cuda:
                light_img = light_img.cuda()
                dark_img = dark_img.cuda()
                # dark_rgb = dark_rgb.cuda()

            light_img = light_img.float()
            # dark_rgb = dark_rgb.float()

            # It's very good practice to keep track of preparation time and
            # computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()

            # forward and backward pass
            optimizer.zero_grad()

            output = net(dark_img)

            # output = (output / (torch.max(output) - torch.min(output))) * 255
            # output = output.clamp(0, 255)
            # output: torch.Tensor = torch.add(output, dark_rgb)
            # output = (output / (torch.max(output) - torch.min(output))) * 255
            # print(output[0])

            # out = torch.from_numpy(np.array([im, ref]))
            # utils.save_image(out, f'out/train/train_{epoch}_{i}.png')
            # utils.save_image(output[0], f'out/train/train_{epoch}_{i}.png')
            # print(output.shape)
            # print(light_img.shape)
            loss = criterion_L2(light_img.float(), output.float())
            avg_loss += loss

            loss.backward()
            optimizer.step()

            #         # udpate tensorboardX
            #         writer.add_scalar(..., n_iter)
            #         ...
            #
            # writer.add_images('Reference', light_img, n_iter)
            # compute computation time and *compute_efficiency*

            process_time = start_time - time.time() - prepare_time
            ce = process_time / (process_time + prepare_time)
            cetotal += ce
            cenum += 1
            pbar.set_description("C/E: {:.2f}, Loss: {:03.3f}, Epoch: {}/{}:".format(
                cetotal / cenum, (avg_loss / (i + 1)), epoch, args.epochs))

            output_grid = utils.make_grid(output, normalize=True, scale_each=True)
            reference_grid = utils.make_grid(light_img, normalize=True, scale_each=True)

            if i == len(train_data_loader) - 1:
                tb_step = epoch
                writer.add_image('Output', output_grid, tb_step)
                writer.add_image('Reference', reference_grid, tb_step)
                writer.add_scalar('loss', avg_loss / (i + 1), tb_step)

            #     writer.add_images('Ouptput', torch.cat((light_img, output), dim=2), epoch)
            # utils.save_image(light_img, f'out/train/train_{epoch}_{i}_ref.png', normalize=True, scale_each=True)
            # utils.save_image(dark_rgb, f'out/train/train_{epoch}_{i}_orig.png', normalize=True, scale_each=True)
            # utils.save_image(output, f'out/train/train_{epoch}_{i}.png', normalize=True, scale_each=True)

            # im: Image = transforms.ToPILImage()(output[0].cpu())
            # im.save(f'out/train/train_{epoch}_{i}_S.png')
            # ref: Image = transforms.ToPILImage()(light_img[0].cpu())
            # ref.save(f'out/train/train_{epoch}_{i}_ref.png')

            start_time = time.time()

        torch.save({
            'epoch': epoch,
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
        }, f'checkpoint/saved_latest.ckpt')
        # maybe do a test pass every x epochs
        x = -1
        if epoch % x == x - 1:
            # torch.cuda.empty_cache()
            # bring models to evaluation mode
            net.eval()
            # do some tests
            pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader)),
                        total=10)
            pbar.set_description("Saving test outputs")
            for i, data in pbar:
                if i == 10:
                    break
                light_img = data['light'].float()
                dark_img = data['dark']
                # dark_rgb = data['dark_rgb'].float()
                if use_cuda:
                    light_img = light_img.cuda()
                    dark_img = dark_img.cuda()
                    # dark_rgb = dark_rgb.cuda()

                output = net(dark_img)
                utils.save_image(light_img, f'out/test/test_{epoch}_{i}_ref.png', normalize=True, scale_each=True)
                # utils.save_image(dark_rgb, f'out/test/test_{epoch}_{i}_orig.png', normalize=True, scale_each=True)
                utils.save_image(output, f'out/test/test_{epoch}_{i}.png', normalize=True, scale_each=True)
