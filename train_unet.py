import torch
import tqdm
from torchvision import transforms
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import MFTDataset_77
from unet import UNetModel
from diffusion import Diffusion
from utils import AvgrageMeter, show_img

batch_size = 200
patch_size = 16
select_spectral = []
spe = 200
channel = 1  # 3d channel

epochs = 1001  # Try more!
lr = 1e-4
T = 500

rgb = [30, 50, 90]
path_prefix = "./save_model/"

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_by_imgs(imgs, rgb=[1, 100, 199]):
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize=(12, 8))
        for j in range(len(imgs)):
            plt.subplot(1, len(imgs), j + 1)
            img = imgs[j][i, 0, rgb, :, :]
            show_img(img)
        plt.show()


def plot_by_images_v2(imgs, rgb=[1, 100, 199]):
    '''
    input image shape is (spectral, height, width)
    '''
    assert len(imgs) > 0
    s, h, w = imgs[0].shape
    plt.figure(figsize=(12, 8))
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j + 1)
        img = imgs[j][rgb, :, :]
        show_img(img)
    plt.show()


def plot_spectral(x0, recon_x0, num=3):
    '''
    x0, recon_x0 shape is (batch, channel, spectral, h, w)
    '''
    batch, c, s, h, w = x0.shape
    step = h // num
    plt.figure(figsize=(20, 5))
    for ii in range(num):
        i = ii * step
        x0_spectral = x0[0, 0, :, i, i]
        recon_x0_spectral = recon_x0[0, 0, :, i, i]
        plt.subplot(1, num, ii + 1)
        plt.plot(x0_spectral, label="x0")
        plt.plot(recon_x0_spectral, label="recon")
        plt.legend()
    plt.show()


def recon_all_fig(diffusion, model, splitX, dataloader, big_img_size=[145, 145]):
    '''
    X shape is (spectral, h, w) => (batch, channel=1, 200, 145, 145)
    '''
    # 1. reconstruct
    t = torch.full((1,), diffusion.T - 1, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(torch.from_numpy(splitX.astype('float32')), t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num=5)

    # ---just for test---
    # recon_from_xt.append(torch.from_numpy(splitX.astype('float32')))
    # plot_by_imgs(recon_from_xt, rgb=rgb)

    # ---------

    res_xt_list = []
    for tempxt in recon_from_xt:
        big_xt = dataloader.split_to_big_image(tempxt.numpy())
        res_xt_list.append(big_xt)
    ori_data, _ = dataloader.get_ori_data()
    res_xt_list.append(ori_data)
    plot_by_images_v2(res_xt_list, rgb=rgb)


def sample_by_t(diffusion, model, x0):
    num = 2
    choose_index = [3]
    # x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    step = diffusion.T // num
    errors = []
    for ti in range(10, diffusion.T, step):
        t = torch.full((1,), ti, device=device, dtype=torch.long)
        xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
        _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num=5)
        recon_x0 = recon_from_xt[-1]
        errors.append(torch.abs(recon_x0 - x0).mean().detach().cpu().numpy())
        # recon_from_xt.append(x0)
        print('---', ti, '---')
        # plot_by_imgs(recon_from_xt, rgb=rgb)
        # plot_spectral(x0, recon_x0)
    return errors


def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 5
    step = all_size // num
    r, g, b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index, :, :, :, :]).float()

    use_t = 499
    # from xt
    t = torch.full((1,), use_t, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num=10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt, rgb=rgb)

    # from noise
    t = torch.full((1,), use_t, device=device, dtype=torch.long)
    _, recon_from_noise = diffusion.reconstruct(model, xt=x0, tempT=t, num=10, from_noise=True, shape=x0.shape)
    plot_by_imgs(recon_from_noise, rgb=rgb)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("save model done. path=%s" % path)


def train(name, path_prefix):
    dataset = MFTDataset_77('houston2013', '77')
    test_dataset = MFTDataset_77('houston2013', '77', 'test')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    diffusion = Diffusion(T=T)
    patch_size = 8
    if name == 'hsi':
        spe = 144
    else:
        spe = 21
    # spe = 144 + 1
    model = UNetModel(
        image_size=patch_size,
        in_channels=spe,
        model_channels=128,
        out_channels=spe,
        num_res_blocks=2,
        attention_resolutions={4},
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        channel_mult=(1, 2,),
        dropout=0.0,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    )

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_metric = AvgrageMeter()

    os.makedirs(path_prefix, exist_ok=True)

    for epoch in tqdm.tqdm(range(epochs)):
        loss_metric.reset()
        for step, (batch, batch2, *_) in enumerate(train_loader):
            batch = transforms.functional.resize(batch.to(device), 8)
            batch2 = transforms.functional.resize(batch2.to(device), 8)
            if name == 'hsi':
                x_0 = batch.to(device)
            else:
                x_0 = batch2.to(device)
            optimizer.zero_grad()
            cur_batch_size = batch.shape[0]
            # print(x_0.shape)
            t = torch.randint(0, diffusion.T, (cur_batch_size,), device=device).long()
            loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, x_0, t)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss.item(), batch.shape[0])

        if epoch % 200 == 0 and epoch > 0:
            model.eval()
            errors_t = []
            for step, (batch, batch2, *_) in enumerate(test_loader):
                batch = transforms.functional.resize(batch.to(device), 8)
                batch2 = transforms.functional.resize(batch2.to(device), 8)
                with torch.no_grad():
                    if name == 'hsi':
                        x_0 = batch.to(device)
                    else:
                        x_0 = batch2.to(device)
                    e = sample_by_t(diffusion, model, x_0)
                    errors_t.append(e)
            errors_t = np.array(errors_t)
            print(f"epoch: {epoch}: ", np.mean(errors_t, axis=0))
            if epoch >= 500:
                path = "%s/unet2d_%s.pth" % (path_prefix, epoch)
                save_model(model, path)


if __name__ == "__main__":
    train('hsi', './save_model_hsi1/')
    train('lidar', './save_model_lidar1/')
