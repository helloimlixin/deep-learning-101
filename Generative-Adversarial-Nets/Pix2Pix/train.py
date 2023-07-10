import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, bce, l1loss, disc_scaler, gen_scaler
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # discriminator training
        # about the amp.autocast() context manager, see: https://pytorch.org/docs/stable/amp.html and
        # this blog from WandB: https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
        # this is a context manager that allows us to use mixed precision training, which essentially reduces the
        # chances of getting CUDA out of memory errors by automatically casting the tensors to a smaller memory footprint,
        # for example, from float32 (default in most PyTorch deep models) to float16.
        with torch.cuda.amp.autocast():
            # forward pass
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        opt_disc.zero_grad()
        disc_scaler.scale(D_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        # generator training
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            l1 = l1loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + l1
        opt_gen.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(
        config.DEVICE
    )  # discriminator initialization
    gen = Generator(in_channels=3, features=64).to(
        config.DEVICE
    )  # generator initialization
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )  # discriminator optimizer initialization
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )  # generator optimizer initialization
    bce = nn.BCEWithLogitsLoss()  # binary cross entropy loss initialization
    l1_loss = nn.L1Loss()  # L1 loss initialization

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)  # dataset initialization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )  # dataloader initialization

    # for gradient scaling, see: https://pytorch.org/docs/stable/amp.html and
    # this blog from WandB: https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
    # gradient scaling addresses the problem of underflowing gradients, i.e., when the gradients are too small to be
    # taken into account by the optimizer, which can lead to the model not learning anything. <code>float16</code> tensors
    # for instance, have a smaller memory footprint than <code>float32</code> tensors, but they can also lead to underflowing
    # as they don't have enough precision to represent the gradients.
    gen_scaler = torch.cuda.amp.GradScaler()  # gradient scaler initialization
    disc_scaler = torch.cuda.amp.GradScaler()  # gradient scaler initialization
    val_dataset = MapDataset(root_dir=config.VAL_DIR)  # dataset initialization
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False
    )  # dataloader initialization

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc,
            gen,
            train_loader,
            opt_disc,
            opt_gen,
            bce,
            l1_loss,
            disc_scaler,
            gen_scaler,
        )

        if config.SAVE_MODEL and epoch % 100 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

        print(f"Epoch {epoch} completed")


if __name__ == "__main__":
    main()
