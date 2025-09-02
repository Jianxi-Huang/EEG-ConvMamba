import mne
from utils_smooth import SmoothGradCAM
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import scipy.io
import torch.nn.functional as F
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt
from torch.backends import cudnn
from utils import GradCAM
from EEG_ConvMamba_HGD import EEG_ConvMamba_HGD
cudnn.benchmark = False
cudnn.deterministic = True

# get data
data = np.load('/Dataset/HGD/test/s5_data.npy')
Data = np.expand_dims(data, axis=1)
label = np.load('/Dataset/HGD/test/s5_label.npy')

CLS = 1
z = 0
# classification
for i in range(160):
    if label[i] == CLS:
        z = z + 1
        if z == 1:
            data_one = Data[i, :, :, :]
        if z != 1:
            x = Data[i, :, :, :]
            data_one = np.concatenate((data_one, x), axis=0)

Data = np.expand_dims(data_one, axis=1)
print(np.shape(data))

nSub = 1
target_category = 0  # set the class (class activation mapping)

# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


device = torch.device("cuda")
model = EEG_ConvMamba_HGD().to(device)


model.load_state_dict(torch.load('/Dataset/EEG-Mamba_model_sub4_HGD.pth', map_location=device), False)
target_layers = [model[1]]  # set the target layer


# Smooth Grad-CAM
smooth_grad_cam = SmoothGradCAM(
    model=model,
    target_layers=target_layers,
    use_cuda=False,
    reshape_transform=reshape_transform,
    noise_level=0.1,
    num_samples=30
)


biosemi_montage = mne.channels.make_standard_montage('standard_1005')
index = [27, 29, 31, 33, 39, 43, 49, 51, 53, 55, 28, 30, 32, 38, 40, 42, 44, 50, 52, 54, 108, 109,
    112, 113, 118, 119, 122, 123, 128, 129, 132, 133, 138, 139, 142, 143, 110, 111, 120, 121, 130, 131,
    140, 141]  # HGD channel
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

# calculate Smooth Grad-CAM
all_cam = []
for i in range(40):
    test = torch.as_tensor(Data[i:i + 1, :, :, :], dtype=torch.float32)
    test = torch.autograd.Variable(test.cuda(), requires_grad=True)

    grayscale_cam = smooth_grad_cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]
    all_cam.append(grayscale_cam)

# the mean of all data
test_all_data = np.squeeze(np.mean(Data, axis=0))
mean_all_test = np.mean(test_all_data, axis=1)

# the mean of all cam
test_all_cam = np.mean(all_cam, axis=0)
mean_all_cam = np.mean(test_all_cam, axis=1)

# apply cam on the input data
test_all_cam_resized = F.interpolate(torch.tensor(test_all_cam).unsqueeze(0).unsqueeze(0),
                                     size=test_all_data.shape,
                                     mode='bilinear',
                                     align_corners=False).squeeze(0).squeeze(0).numpy()


hyb_all = test_all_data * test_all_cam_resized
mean_hyb_all = np.mean(hyb_all, axis=1)

mean_all_test = (mean_all_test - np.min(mean_all_test)) / (np.max(mean_all_test) - np.min(mean_all_test))
mean_hyb_all = (mean_hyb_all - np.min(mean_hyb_all)) / (np.max(mean_hyb_all) - np.min(mean_hyb_all))

evoked = mne.EvokedArray(test_all_data, info)
evoked.set_montage('standard_1005')

fig, [ax1, ax2] = plt.subplots(nrows=2)

im1, cn = mne.viz.plot_topomap(mean_all_test, evoked.info, show=False, axes=ax1, res=1200, cmap='coolwarm')
plt.colorbar(im1)

im2, cn2 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax2, res=1200, cmap='coolwarm')
plt.colorbar(im2)


plt.savefig('/Dataset/results/Smooth grad cam.png', dpi=600)
plt.show()
