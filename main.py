import sys
import torch
import os
import torch.nn as nn
import torch.optim as optim
from data_preprocess import Preprocess_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as transforms

TRAIN_DIR = "Project7_CycleGAN/data/train"
TEST_DIR = "Project7_CycleGAN/data/test"
SAVE_DIR = "Project7_CycleGAN/save_images"
LOAD_MODEL = False
MODEL_GEN_B = "gen_b.wyd"
MODEL_GEN_A = "gen_a.wyd"
MODEL_DISC_A = "dis_a.wyd"
MODEL_DISC_B = "dis_b.wyd"
MODEL_OPT_GEN = "opt_a.wyd"
MODEL_OPT_DIS = "opt_b.wyd"
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.Identity(),
        )   
    
    def forward(self, x):
        return x + self.residual(x)

#Discrminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect"),
        )

    def call(self, image):
        image = self.block(image)
            
        return torch.sigmoid(image)

#Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
        )

    def call(self, image):
        image = self.block(image)
        return torch.tanh(image)

L1_Loss = nn.L1Loss()
MSE_Loss = nn.MSELoss()

def Generator_Loss(real_A, real_B, fake_A, fake_B, Gen_A, Gen_B, Dis_A, Dis_B):
    # Identity loss
    ##################################################
    identity_B = Gen_B.call(real_B)

    identity_B_loss = L1_Loss(real_B, identity_B)

    identity_A = Gen_A.call(real_A)

    identity_A_loss = L1_Loss(real_A, identity_A)

    # Adversial loss
    ##################################################
    Dis_A_fake = Dis_A.call(fake_A)

    loss_Gen_A = MSE_Loss(Dis_A_fake, torch.ones_like(Dis_A_fake))

    Dis_B_fake = Dis_B.call(fake_B)

    loss_Gen_B = MSE_Loss(Dis_B_fake, torch.ones_like(Dis_B_fake))

    # Cycle loss
    ##################################################
    cycle_B = Gen_B.call(fake_A)

    cycle_B_loss = L1_Loss(real_B, cycle_B)

    cycle_A = Gen_A.call(fake_B)

    cycle_A_loss = L1_Loss(real_A, cycle_A)

    # Total loss
    ##################################################
    Gen_loss = loss_Gen_B + loss_Gen_A + cycle_B_loss * 10 + cycle_A_loss * 10 + identity_A_loss * 0.5 + identity_B_loss * 0.5
    return Gen_loss


def Discriminator_Loss(real_A, real_B, fake_A, fake_B, Dis_A, Dis_B):
    #Real Loss
    ##################################################
    Dis_A_real = Dis_A.call(real_A)

    Dis_B_real = Dis_B.call(real_B)

    Dis_A_real_loss = MSE_Loss(Dis_A_real, torch.ones_like(Dis_A_real))

    Dis_B_real_loss = MSE_Loss(Dis_B_real, torch.ones_like(Dis_B_real))

    #Fake Loss
    ##################################################
    Dis_A_fake = Dis_A.call(fake_A.detach())

    Dis_A_fake_loss = MSE_Loss(Dis_A_fake, torch.zeros_like(Dis_A_fake))

    Dis_B_fake = Dis_B.call(fake_B.detach())

    Dis_B_fake_loss = MSE_Loss(Dis_B_fake, torch.zeros_like(Dis_B_fake))

    #Total Loss
    ##################################################
    Dis_real_loss = Dis_A_real_loss + Dis_B_real_loss

    Dis_fake_loss = Dis_A_fake_loss + Dis_B_fake_loss

    Dis_loss = Dis_real_loss + Dis_fake_loss
    return Dis_loss


#convert imgae into tensor
transforms_ = [ transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]

def test_generator():
    x = torch.randn((2, 3, 256, 256))
    gen = Generator()
    print(gen.call(x).shape)

def test_discriminator():
    x = torch.randn((2, 3, 256, 256))
    dis = Discriminator()
    print(dis.call(x).shape)

def main():
    #Preprocess the data
    train_dataset = Preprocess_Dataset(dir_A=TRAIN_DIR+"/monet", dir_B=TRAIN_DIR+"/photos", convertor=transforms_)
    test_dataset = Preprocess_Dataset(dir_A=TEST_DIR+"/monet", dir_B=TEST_DIR+"/photos", convertor=transforms_)
    train_dataset = DataLoader(train_dataset,batch_size=3,shuffle=True)
    test_dataset = DataLoader(test_dataset,batch_size=1,shuffle=False)

    #initialize discriminator and generator
    Gen_B = Generator().to("cuda")
    Gen_A = Generator().to("cuda")
    Dis_A = Discriminator().to("cuda")
    Dis_B = Discriminator().to("cuda")

    #initialize optimizer and try to load discrminator and generator
    optimizer_gen = optim.Adam(list(Gen_B.parameters()) + list(Gen_A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(list(Dis_A.parameters()) + list(Dis_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

    if LOAD_MODEL:
        Dis_A.load_state_dict(torch.load(MODEL_DISC_A))
        Dis_A.eval()

        Dis_B.load_state_dict(torch.load(MODEL_DISC_B))
        Dis_B.eval()

        Gen_A.load_state_dict(torch.load(MODEL_GEN_A))
        Gen_A.eval()

        Gen_B.load_state_dict(torch.load(MODEL_GEN_B))
        Gen_B.eval()

        optimizer_gen.load_state_dict(torch.load(MODEL_OPT_GEN))

        optimizer_disc.load_state_dict(torch.load(MODEL_OPT_DIS))

    for epoch in range(10):
        print('\r', )
        A_reals = 0
        A_fakes = 0
        print("current epoch:", epoch + 1)
        DATA = tqdm(train_dataset, leave=True)

        for i, (B, A) in enumerate(DATA):
            real_A = A.to("cuda")
            real_B = B.to("cuda")

            fake_A = Gen_A.call(real_B)
            fake_B = Gen_B.call(real_A)
            #loss function of generator
            optimizer_gen.zero_grad()
            with torch.cuda.amp.autocast():
                Gen_loss = Generator_Loss(real_A, real_B, fake_A, fake_B, Gen_A, Gen_B, Dis_A, Dis_B)
                Gen_loss.backward()
                optimizer_gen.step()


            #loss function of discrminator
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast():
                Dis_loss = Discriminator_Loss(real_A, real_B, fake_A, fake_B, Dis_A, Dis_B)
                Dis_loss.backward()
                optimizer_disc.step()

                #update mean of A_reals and A_fakes
                ##################################################
                Dis_A_real = Dis_A.call(real_A)
                Dis_A_fake = Dis_A.call(fake_A.detach())

                A_reals += Dis_A_real.mean().item()
                A_fakes += Dis_A_fake.mean().item()

            DATA.set_postfix(A_real=A_reals/(i+1), A_fake=A_fakes/(i+1))

        torch.save(Dis_A.state_dict(), MODEL_DISC_A)
        torch.save(Dis_B.state_dict(), MODEL_DISC_B)
        torch.save(Gen_B.state_dict(), MODEL_GEN_B)
        torch.save(Gen_A.state_dict(), MODEL_GEN_A)
        torch.save(optimizer_gen.state_dict(), MODEL_OPT_GEN)
        torch.save(optimizer_disc.state_dict(), MODEL_OPT_DIS)
 
    if not os.path.exists('output/'):
        os.makedirs('output/')


    DATA = tqdm(test_dataset, leave=True)
    for i, (B, A) in enumerate(DATA):
        real_A = A.to("cuda")
        real_B = B.to("cuda")

        fake_A = Gen_A.call(real_B)
        fake_B = Gen_B.call(real_A)

        save_image(fake_A*0.5+0.5, f"output/fake_A_{i}.png")
        save_image(fake_B*0.5+0.5, f"output/fake_B_{i}.png")

if __name__ == "__main__":
    main()