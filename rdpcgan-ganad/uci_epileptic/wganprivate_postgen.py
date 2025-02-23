import argparse
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import dp_optimizer
import analysis
import pandas as pd
import re
from DPAE2 import DP_Autoencoder

# model_epoch_50.pth
# def find_highest_x(folder_path, classlabel):
#     # Pattern to match the files
#     pattern = re.compile(fr'model_epoch_(\d+)_{int(classlabel)}\.pth')
#
#     highest_x = None
#
#     # List all files in the given folder
#     for filename in os.listdir(folder_path):
#         match = pattern.match(filename)
#         if match:
#             x = int(match.group(1))
#             if highest_x is None or x > highest_x:
#                 highest_x = x
#
#     return highest_x

parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]
experimentName = 'uci'

parser.add_argument("--DATASETDIR", type=str,
                    default=os.path.expanduser('~/data/UCI'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=10,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.005, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.00001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent noise space")
parser.add_argument("--feature_size", type=int, default=1071, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between batches")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval between image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=10, help="number of epochs per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=False, help="Minibatch averaging")

#### Privacy
parser.add_argument('--dp_privacy', type=bool, default=True)
parser.add_argument('--noise_multiplier', type=float, default=1)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=1.0)
parser.add_argument('--delta', type=float, default=1e-5, help="Target delta (default: 1e-5)")

# Training/Testing
parser.add_argument("--pretrained_status", type=bool, default=True, help="If want to use ae pretrained weights")
parser.add_argument("--pretrained_anomaly_status", type=bool, default=True, help="If you want to use anomaly detection")
parser.add_argument("--classLabel", type=float, default=1.0, help="Define the class label to train the model to generate")
parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=False, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default='checkpoints_postgen/',
                    help="Model path")
parser.add_argument("--AD_budget", type=float, default=-1, help="the budget for anomaly detection")
parser.add_argument("--use_midgen", type=bool, default=False, help="Use midgen")
opt = parser.parse_args()
print(opt)

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir -p {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


#########################
######## Privacy ########
#########################

def _set_seed(secure_seed: int):
    if secure_seed is not None:
        secure_seed = secure_seed
    else:
        secure_seed = int.from_bytes(
            os.urandom(8), byteorder="big", signed=True
        )
    return secure_seed

# Generate secure seed
secure_seed = _set_seed(None)

# Secure generator
_secure_generator = (
    torch.random.manual_seed(secure_seed)
    if device.type == "cpu"
    else torch.cuda.manual_seed(secure_seed)
)

# Generate noise
def _generate_noise(max_norm, parameter):
    if opt.noise_multiplier > 0:
        return torch.normal(
            0,
            opt.noise_multiplier * max_norm,
            parameter.grad.shape,
            device=device,
            generator=_secure_generator,
        )
    return 0.0


def clip_grads_(model):
    # Calculate norm
    total_norm = 0
    for param in model.parameters():
        if param.requires_grad:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** .5
    clip_val = min(opt.max_per_sample_grad_norm / (total_norm + 1e-6), 1.)

    # Clip grads
    for param in model.parameters():
        if param.requires_grad:
            # in-place multiplication with coefficient
            param.grad.data.mul_(clip_val)


def add_noise_(model):
    # Adding noise
    params = (p for p in model.parameters() if p.requires_grad)
    for p in params:
        noise = _generate_noise(opt.max_per_sample_grad_norm, p)
        p.grad += noise


##########################
### Dataset Processing ###
##########################
# Read data with the last dimension that is the class label
# trainData = pd.read_csv(os.path.join(opt.DATASETDIR,'train.csv')).drop('Unnamed', axis=1).to_numpy()
# testData = pd.read_csv(os.path.join(opt.DATASETDIR,'test.csv')).drop('Unnamed', axis=1).to_numpy()

data = pd.read_csv('uci-epileptic.csv').to_numpy()
data = MinMaxScaler().fit_transform(data)


if opt.AD_budget != -1:
    print("Using AD")

    anomaly_detector = DP_Autoencoder(data.shape[1], int(data.shape[1] / 2), epsilon=opt.AD_budget)
    anomaly_detector.train(data)
    if not opt.use_midgen:
        original_train_indices = anomaly_detector.get_top_anomalies(data, num_anomalies=10)
        data = np.delete(data, original_train_indices, axis=0)


train_ratio = 0.5
train = np.random.rand(data.shape[0]) < train_ratio
train_data, test_data = data[train], data[~train]
data_dim = data.shape[1]


# Class specific data
classLabel = opt.classLabel
trainData = train_data[train_data[:,-1] == classLabel]

class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            pass

        return torch.from_numpy(sample)


# Train data loader
dataset_train_object = Dataset(data=trainData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

# Test data loader
dataset_test_object = Dataset(data=test_data, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=True, num_workers=0, drop_last=True)

# Generate random samples for test
random_samples = next(iter(dataloader_test))
feature_size = random_samples.size()[1]

###########################
## Privacy Calculation ####
###########################
if opt.dp_privacy:
    totalsamples = len(dataset_train_object)
    num_batches = len(dataloader_train)
    iterations = opt.n_epochs * num_batches
    print(f"{totalsamples=}")
    print(f"{opt.batch_size=}")
    print(f"{opt.noise_multiplier=}")
    print(f"{iterations=}")
    print(f"{opt.delta=}")
    print('Achieves ({}, {})-DP'.format(
            analysis.epsilon(
                totalsamples,
                opt.batch_size,
                opt.noise_multiplier,
                iterations,
                opt.delta
            ),
            opt.delta,
        ))
####################
### Architecture ###
####################
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=3, stride=1,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5,
                               stride=1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5,
                               stride=4, padding=0,
                               dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7,
                               stride=4,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=1, kernel_size=7, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        return torch.squeeze(x, dim=1)

    def decode(self, x):
        x = self.decoder(x)
        return torch.squeeze(x, dim=1)

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        return torch.squeeze(x, dim=1)

    def decode(self, x):
        x = self.decoder(x)
        return torch.squeeze(x, dim=1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 4
        self.main = nn.Sequential(
        nn.ConvTranspose1d(opt.latent_dim, ngf * 16, 4, 1, 0),
        nn.BatchNorm1d(ngf * 16, eps=0.0001, momentum=0.01),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1),
        nn.BatchNorm1d(ngf * 8, eps=0.0001, momentum=0.01),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
        nn.BatchNorm1d(ngf * 4, eps=0.0001, momentum=0.01),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1),
        nn.BatchNorm1d(ngf * 2, eps=0.0001, momentum=0.01),
        nn.LeakyReLU(0.2, inplace=True),
        # nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1),
        # nn.BatchNorm1d(ngf, eps=0.001, momentum=0.01),
        # nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose1d(ngf * 2, 1, 4, 2, 1),
        nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, x.shape[1], 1)
        out = self.main(x)
        return torch.squeeze(out, dim=1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 16
        self.conv1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(1, ndf, 8, 4, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv1d(ndf, ndf * 2, 8, 4, 1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 8, 4, 1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.conv4 = nn.Sequential(
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv1d(ndf * 4, ndf * 8, 8, 4, 1),
        #     nn.BatchNorm1d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        self.conv4 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 4, 1, 2, 1, 0),
            # nn.Sigmoid()
        )

    def forward(self, input):
        out = self.conv1(input.view(-1, 1, input.shape[1]))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return torch.squeeze(out, dim=2)

###############
### Lossess ###
###############

def _gradient_penalty(real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if self.use_cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if self.use_cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return self.gp_weight * ((gradients_norm - 1) ** 2).mean()


# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def calc_gradient_penalty(netD, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def autoencoder_loss(x_output, y_target):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the latter one, mean over both features and batches.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


#################
### Functions ###
#################

def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def sample_transform(sample):
    """
    Transform samples to their nearest integer
    :param sample: Rounded vector.
    :return:
    """
    sample[sample >= 0.5] = 1
    sample[sample < 0.5] = 0
    return sample


def weights_init(m):
    """
    Custom weight initialization.
    NOTE: Bad initialization may lead to dead model and can prevent training!
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize generator and discriminator
discriminatorModel = Discriminator()
generatorModel = Generator()
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder
autoencoder_anomaly_Model = Autoencoder()


# Define cuda Tensors
# BE careful about torch.FloatTensor([1])!!!!
# I once defined it as torch.FloatTensor(1) without brackets around 1 and everything was messed hiddenly!!
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

if torch.cuda.device_count() > 1 and opt.multiplegpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    generatorModel = nn.DataParallel(generatorModel, list(range(opt.num_gpu)))
    discriminatorModel = nn.DataParallel(discriminatorModel, list(range(opt.num_gpu)))
    autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
    autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))
    autoencoder_anomaly_Model = nn.DataParallel(autoencoder_anomaly_Model, list(range(opt.num_gpu)))

if opt.cuda:
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    generatorModel.cuda()
    discriminatorModel.cuda()
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    autoencoder_anomaly_Model.cuda()

    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
autoencoderModel.apply(weights_init)
autoencoder_anomaly_Model.apply(weights_init)

# Optimizers
g_params = [{'params': generatorModel.parameters()},
            {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]
# g_params = list(generatorModel.parameters()) + list(autoencoderModel.decoder.parameters())
optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

if opt.dp_privacy:
    optimizer_D = dp_optimizer.AdamDP(
            max_per_sample_grad_norm=opt.max_per_sample_grad_norm,
            noise_multiplier=opt.noise_multiplier,
            batch_size=opt.batch_size,
            params=discriminatorModel.parameters(),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            weight_decay=0.0001,
        )

    optimizer_A = dp_optimizer.AdamDP(
            max_per_sample_grad_norm=opt.max_per_sample_grad_norm,
            noise_multiplier=opt.noise_multiplier,
            batch_size=opt.batch_size,
            params=autoencoderModel.parameters(),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            weight_decay=0.0001,
        )
    optimizer_anomaly_detection = dp_optimizer.AdamDP(
            max_per_sample_grad_norm=opt.max_per_sample_grad_norm,
            noise_multiplier=opt.noise_multiplier,
            batch_size=opt.batch_size,
            params=autoencoderModel.parameters(),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            weight_decay=0.0001,
        )

else:
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)
    optimizer_A = torch.optim.Adam(autoencoderModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)
    optimizer_anomaly_detection =  torch.optim.Adam(autoencoderModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

################
### TRAINING ###
################
if opt.training:

    #@todo: add support for resume with anomaly detection
    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.modelPATH, "model_epoch_50.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load epoch number
        epoch_resume = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()

    else:
        epoch_resume = 0
        print('Training from scratch...')

    if not opt.pretrained_status:
        print('No pretrained autoencoder')
        sys.exit()
    else:
        print('loading pretrained autoencoder...')

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.modelPATH, "aepretrained.pth"))

        # Load models
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])

        # Load optimizers
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load weights
        autoencoderModel.eval()

    # if not opt.pretrained_anomaly_status:
    #     print('No pretrained anomaly detection')
    #     sys.exit()
    # else:
    #     print('loading pretrained anomaly detection...')
    #
    #     # Loading the checkpoint
    #     checkpoint = torch.load(os.path.join(opt.modelPATH, "ae_anomaly_pretrained.pth"))
    #
    #     # Load models
    #     autoencoder_anomaly_Model.load_state_dict(checkpoint['autoencoder_anomaly_detection_state_dict'])
    #
    #     # Load optimizers
    #     optimizer_anomaly_detection.load_state_dict(checkpoint['optimizer_anomaly_detection_state_dict'])
    #
    #     # Load weights
    #     autoencoder_anomaly_Model.eval()




    gen_iterations = 0
    for epoch in range(epoch_resume, opt.n_epochs):
        epoch_start = time.time()
        for i_batch, samples in enumerate(dataloader_train):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # reset gradients of discriminator
            optimizer_D.zero_grad()

            if opt.dp_privacy:

                # Microbatch processing
                for i in range(opt.batch_size):
                    # Extract microbatch
                    micro_batch = real_samples[i:i + 1, :]

                    for p in discriminatorModel.parameters():  # reset requires_grad
                        p.requires_grad = True

                        # @todo: implement midgen
                        if opt.use_midgen:
                            #clean microbatch
                            original_train_indices = anomaly_detector.get_top_anomalies(micro_batch.numpy(), num_anomalies=1)
                            micro_batch = np.delete(micro_batch, original_train_indices, axis=0)


                    # Error on real samples
                    errD_real = torch.mean(discriminatorModel(micro_batch), dim=0)
                    errD_real.backward(one)

                    # Measure discriminator's ability to classify real from generated samples
                    # The detach() method constructs a new view on a tensor which is declared
                    # not to need gradients, i.e., it is to be excluded from further tracking of
                    # operations, and therefore the subgraph involving this view is not recorded.
                    # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                    # Sample noise as generator input
                    z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                    # Generate a batch of images
                    fake = generatorModel(z)

                    fake_decoded = torch.squeeze(autoencoderDecoder(fake.unsqueeze(dim=2)), dim=1)

                    if opt.use_midgen:
                        original_train_indices = anomaly_detector.get_top_anomalies(fake_decoded, num_anomalies=1)
                        micro_batch = np.delete(fake_decoded, original_train_indices, axis=0)
                        pass



                    errD_fake = torch.mean(discriminatorModel(fake_decoded.detach()), dim=0)
                    errD_fake.backward(mone)  # Err_D fake seems to be multiplied by -1 and then substracted

                    # Error


                    # reconstructed_fake = autoencoder_anomaly_Model(fake_decoded.detach())


                    # reconstruction_error_fake = torch.mean((reconstructed_fake - fake_decoded) ** 2) #this error is small so hopefully should only modify it in a small manner



                    errD = errD_real - errD_fake
                    # errD += 0.1 * reconstruction_error_fake
                    # # errD.backward(one)

                    # Bounding sensitivity
                    optimizer_D.clip_grads_()

                # Optimizer step
                optimizer_D.add_noise_()
                optimizer_D.step()

            else:
                for p in discriminatorModel.parameters():  # reset requires_grad
                    p.requires_grad = True

                # Error on real samples
                errD_real = torch.mean(discriminatorModel(real_samples), dim=0)
                errD_real.backward(one)

                # Sample noise as generator input
                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                # Generate a batch of images
                fake = generatorModel(z)

                fake_decoded = torch.squeeze(autoencoderDecoder(fake.unsqueeze(dim=2)), dim=1)
                errD_fake = torch.mean(discriminatorModel(fake_decoded.detach()), dim=0)
                errD_fake.backward(mone)

                # Error

                errD = errD_real - errD_fake
                # errD.backward(one)

                optimizer_D.step()

            # clamp parameters to a cube
            for p in discriminatorModel.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # -----------------
            #  Train Generator
            # -----------------

            # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
            #
            # In PyTorch, we need to set the gradients to zero before starting to do backpropagation because PyTorch
            # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
            # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly. Else the gradient would point in some other direction
            # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

            # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
            # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
            # (additive) accumulator variable is initialized to 0 in code.
            # WARNING. BE VERY CAREFUL about the backward arguments: https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generate a batch of images
            fake = generatorModel(z)

            # uncomment if there is no autoencoder
            fake_decoded = torch.squeeze(autoencoderDecoder(fake.unsqueeze(dim=2)),dim=1)

            # Loss measures generator's ability to fool the discriminator

            errG = torch.mean(discriminatorModel(fake_decoded), dim=0)
            errG.backward(one)

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

            batches_done = epoch * len(dataloader_train) + i_batch + 1
            if batches_done % opt.sample_interval == 0:
                print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.6f Loss_G: %.6f Loss_D_real: %.6f Loss_D_fake %.6f'
                      % (epoch + 1, opt.n_epochs, i_batch + 1, len(dataloader_train),
                         errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)

        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0 or (epoch+1) == (opt.n_epochs + epoch_resume) :
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
            }, os.path.join(opt.modelPATH, "model_epoch_%d_%d.pth" % (epoch + 1, int(classLabel))))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            # call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)
