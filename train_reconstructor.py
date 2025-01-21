import time
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from utils.dataloader import *
from model.Net1 import ReconNet
import torch
import os
from tensorboardX import SummaryWriter
  

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--model_name', type=str, default='Esti_Net1_c48_b10_p64_tune1')
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")
parser.add_argument('--layer_num', type=int, default=10, help='number of epochs to train')
parser.add_argument('--channel', type=int, default=48, help='number of epochs to train')
parser.add_argument('--load_pretrain', type=bool, default=True)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--patch_size', type=int, default=72)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--patchsize_train', type=int, default=72, help='patchsize of LR images for training')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_epoch', type=int, default=10, help='number of epochs to save')
parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.75, help='learning rate decaying factor')
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
parser.add_argument('--trainset_dir', type=str, default='/home/ustc/Code/xiaozy/BlindLFSR/data4blindLFSR/train.h5')
parser.add_argument('--testset_dir', type=str, default='/home/ustc/Code/xiaozy/BlindLFSR/data4blindLFSR/valid/')

args = parser.parse_args()

def getNetworkDescription(network):
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n


def save_ckpt(args, net, idx_epoch):
    if args.parallel:
        torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                    '/home/ustc/Code/xiaozy/BlindLFSR/codes/log/' + args.model_name + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')
    else:
        torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                    '/home/ustc/Code/xiaozy/BlindLFSR/codes/log/' + args.model_name + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')
    print('/home/ustc/Code/xiaozy/BlindLFSR/codes/log/' + args.model_name + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')

def train(args):
    net = ReconNet(channel=args.channel, factor=args.upfactor, angRes=args.angRes,\
                   ksize=21,layer=args.layer_num,\
                    kernel_pretrain='/home/ustc/Code/xiaozy/BlindLFSR/codes/log/DegEst2_5_4xSR.tar')
    net.cuda()
    _,paras = getNetworkDescription(net)
    print(paras)
    cudnn.benchmark = True
    epoch_state = 0
    writer = SummaryWriter(log_dir="{}tb/{}/".format('/home/ustc/Code/xiaozy/BlindLFSR/codes/log/', args.model_name),
                               comment="Training curve for LFASR")
    if args.load_pretrain:
        if os.path.isfile(args.model_path):
            model = torch.load(args.model_path, map_location={'cuda:0': args.device})
            net.load_state_dict(model['state_dict'], strict=True)
            epoch_state = model["epoch"]
            print('laoded pretrained')
        else:
            print("=> no model found at '{}'".format(args.load_model))

    if args.parallel:
        net = torch.nn.DataParallel(net)

    criterion_Loss = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []

    for idx_epoch in range(epoch_state, args.n_epochs):
        torch.cuda.empty_cache()
        gen_LR = MultiDegrade(
            scale=args.upfactor,
            kernel_size=21,
            sig_min=0,
            sig_max=4,
        )

        train_set = TrainSetLoader(args)
        train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
        
        kernel_size = 21
        
        for idx_iter, lf_hr in enumerate(train_loader,1):
        # for i, batch in enumerate(train_loader, 1):
            lfimg_hr = rearrange(lf_hr, 'b u v c h w -> b (u v) c h w')
            lfimg_lr, kernels, sigma, noise_level = gen_LR(lfimg_hr.cuda())
            lf_lr = rearrange(lfimg_lr, 'b (u v) c h w -> b u v c h w', u=args.angRes, v=args.angRes)
            

            ker_map = torch.unsqueeze(kernels,1)
            ker_map = torch.unsqueeze(ker_map,1)
            # print(kernels.shape)
            ker_map = ker_map.expand((kernels.shape[0], args.angRes, args.angRes, kernel_size, kernel_size))
            ker_map = ker_map.reshape(kernels.shape[0], args.angRes, args.angRes, kernel_size*kernel_size)
            # print(ker_map.shape)    # torch.Size([8, 5, 5, 441])


            bdr = 12 // args.upfactor
            label, data = lf_hr[:, :, :, :, 12:-12, 12:-12], lf_lr[:, :, :, :, bdr:-bdr, bdr:-bdr]
            label, data = augmentation(label, data)
            gt_blur = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes, args.angRes) / 4

            optimizer.zero_grad()
            srs = net(data)
            loss = 0

            loss = criterion_Loss(srs, label.cuda())
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        writer.add_scalar("train/all_loss", loss, idx_epoch)
        writer.flush()

        print(time.ctime()[4:-5] + ' Epoch----%5d, loss_pixel---%f, lr----%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()),optimizer.state_dict()['param_groups'][0]['lr']))
        if idx_epoch % args.save_epoch == 0:
            save_ckpt(args, net, idx_epoch+1)

        ''' evaluation '''
        if idx_epoch % args.save_epoch == 0:
            for noise in [0, 15, 50]:
                args.noise = noise
                for sig in [0, 1.5, 3]:
                    args.sig = sig
                    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
                    for index, test_name in enumerate(test_Names):
                        torch.cuda.empty_cache()
                        test_loader = test_Loaders[index]
                        psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                        print('Dataset--%15s,\t noise--%f, \t sig---%f, \t PSNR--%f, \t SSIM---%f' % (
                        test_name, args.noise, sig, psnr_epoch_test, ssim_epoch_test))
                        txtfile = open('/home/ustc/Code/xiaozy/BlindLFSR/codes/log/' + args.model_name + '_training_log.txt', 'a')
                        txtfile.write('Epoch--%f,\t Dataset--%15s,\t noise--%f,\t sig--%f,\t PSNR---%f,\t SSIM---%f \n' % (
                            idx_epoch + 1, test_name, args.noise, sig, psnr_epoch_test, ssim_epoch_test))
                        txtfile.close()

            txtfile = open('/home/ustc/Code/xiaozy/BlindLFSR/codes/log/' + args.model_name + '_training_log.txt', 'a')
            txtfile.write('\n')
            txtfile.close()

        scheduler.step()


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label, sigma, noise_level) in (enumerate(test_loader)):

        gt_blur = sigma.unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes, args.angRes) / 4
        gt_noise = noise_level.repeat(1, 1, args.angRes, args.angRes)

        if args.crop == False:
            with torch.no_grad():
                outLF = net(data)
                outLF = outLF.squeeze()
        else:
            patch_size = args.patchsize_test
            data = data.squeeze()
            sub_lfs = LFdivide(data, patch_size, patch_size // 2)

            # print(sub_lfs.shape)
            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            mini_batch = args.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    lf_out = net(input_lfs.cuda())
                    out_lfs.append(lf_out)
                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[(idx_inference+1) * mini_batch :, :, :, :, :, :]
                    lf_out = net(input_lfs.cuda())
                    out_lfs.append(lf_out)

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * args.upfactor, patch_size * args.upfactor // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * args.upfactor, 0 : data.shape[4] * args.upfactor]


        psnr, ssim = cal_metrics(label.squeeze(), outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        x = torch.flip(x, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        y = torch.flip(y, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x, y = x[:, :, :, color, :, :], y[:, :, :, color, :, :]

    return x, y


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train(args)
