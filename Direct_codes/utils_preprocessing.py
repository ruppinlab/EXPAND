import numpy as np, cv2, torch
from torchvision.models import resnet50, ResNet50_Weights


#### ----------------------------------------------------------------
#%%  init functions.
#### ----------------------------------------------------------------

def set_random_state(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def set_device(verbose = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\ndevice = {device}")
    return device


#### ----------------------------------------------------------------
#%%  ResNet50 for feature extraction.
#### ----------------------------------------------------------------

class ResNetModel(torch.nn.Module):
    def __init__(self, model_type = "load_from_saved_file"):
        super().__init__()
        
        if model_type == "load_from_internet":
            self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        elif model_type == "load_from_saved_file":
            self.resnet = resnet50(weights = None)
        else:
            raise ValueError("cannot find model_type! can only be 'load_from_internet' or 'load_from_saved_file'")
    
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

##======================================================================================================
def evaluate_tile(img_rgb, edge_mag_thrsh, edge_fraction_thrsh):
    ## convert from rgb to grayscale.
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    ## remove noise using a Gaussian filter.
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    ## detect edges using Sobel gradien operator.
    dx_sobel = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    dy_sobel = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    
    ## get absolute values of gradients & compute weighted sum (weight = 0.5).
    dx_sobel_val = cv2.convertScaleAbs(dx_sobel)
    dy_sobel_val = cv2.convertScaleAbs(dy_sobel)
    mag = cv2.addWeighted(dx_sobel_val, 0.5, dy_sobel_val, 0.5, 0)
    
    ## exclude tile if edge_mag > 0.5 (select = 0).
    unique, counts = np.unique(mag, return_counts = True)
    # img_area = np.prod(img_rgb.shape)            # returns LinAlgError: eigenvalues did not converge
    img_area = img_rgb.shape[0] ** 2
    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum() / img_area
    select   = int(not(edge_mag > edge_fraction_thrsh))
    
    return select

