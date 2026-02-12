import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from medpy.metric import jc
import logging
# import nibabel as nib

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

seed = 66
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)

    def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
        return np.argmax(onehot_output, axis=-1)

    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized

    def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
        return np.argmax(onehot_output, axis=-1)


    def deformation_to_transformation(dx, dy):

        nx, ny = dx.shape

        # grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
        grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # Robin's change to make it work with non-square images

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        return map_x, map_y

    def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR, do_optimisation=True):

        map_x, map_y = deformation_to_transformation(dx, dy)

        # The following command converts the maps to compact fixed point representation
        # this leads to a ~20% increase in speed but could lead to accuracy losses
        # Can be uncommented
        if do_optimisation:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)
        return cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))


    def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR, do_optimisation=True):

        onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp, do_optimisation=do_optimisation)
        return np.argmax(onehot_output, axis=-1)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s + 1e-6)


def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)


def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)


def generalised_energy_distance(sample_arr, gt_arr, nlabels, **kwargs):

    def dist_fct(m1, m2):

        label_range = kwargs.get('label_range', range(nlabels))

        per_label_iou = []
        for lbl in label_range:

            # assert not lbl == 0  # tmp check
            m1_bin = (m1 == lbl)*1
            m2_bin = (m2 == lbl)*1

            if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                per_label_iou.append(1)
            elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                per_label_iou.append(0)
            else:
                per_label_iou.append(jc(m1_bin, m2_bin))

        # print(1-(sum(per_label_iou) / nlabels))

        return 1-(sum(per_label_iou) / nlabels)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i,...], gt_arr[j,...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i,...], sample_arr[j,...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i,...], gt_arr[j,...]))

    return (2./(N*M))*sum(d_sy) - (1./N**2)*sum(d_ss) - (1./M**2)*sum(d_yy)

def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=0)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
    sample_arr = sample_arr.detach().cpu().numpy()
    gt_arr = gt_arr.detach().cpu().numpy()

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[2]
    sY = sample_arr.shape[3]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)


def show_tensor(tensor):
    """Show images with matplotlib for debugging, only for 128,128"""
    with torch.no_grad():
        import matplotlib.pyplot as plt
        try:
            tensor = tensor.detach()
        except:
            pass

        height = tensor.shape[2]
        width = tensor.shape[3]

        batch_size = tensor.shape[0]

        result = tensor[0].view(height, width)
        for i in range(1, batch_size):
            result = torch.cat([result, tensor[i].view(height, width)], dim=1)

        plt.imshow(result, cmap='Greys_r')

# def convert_to_onehot(lblmap, nlabels):
#
#     output = torch.zeros((lblmap.shape[0], nlabels, lblmap.shape[2], lblmap.shape[3]))
#     for ii in range(nlabels):
#         output[:, ii, :, :] = (lblmap == ii).view(-1, lblmap.shape[2], lblmap.shape[3]).long()
#
#     assert output.shape == (lblmap.shape[0], nlabels, lblmap.shape[2], lblmap.shape[3])
#
#     return output
def convert_to_onehot(lblmap, nlabels):

    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:, :, ii] = (lblmap == ii).astype(np.uint8)
    return output


# needs a torch tensor as input instead of numpy array
# accepts format HW and CHW
def convert_to_onehot_torch(lblmap, nlabels):
    # print('labelmap shape:', lblmap.shape)
    if len(lblmap.shape) == 3:
        # 2D image
        output = torch.zeros((nlabels, lblmap.shape[-2], lblmap.shape[-1]))
        for ii in range(nlabels):
            lbl = (lblmap == ii).view(lblmap.shape[-2], lblmap.shape[-1])
            output[ii, :, :] = lbl
    elif len(lblmap.shape) == 4:
        # 3D images from brats are already one hot encoded
        output = lblmap
    return output.long()



def convert_batch_to_onehot(lblbatch, nlabels):
    out = []
    for ii in range(lblbatch.shape[0]):
        lbl = convert_to_onehot_torch(lblbatch[ii,...], nlabels)
        # TODO: check change
        out.append(lbl.unsqueeze(dim=0))

    result = torch.cat(out, dim=0)
    return result


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def convert_nhwc_to_nchw(tensor):
    result = tensor.transpose(1, 3).transpose(2, 3)
    return result


def convert_nchw_to_nhwc(tensor):
    result = tensor.transpose(1, 3).transpose(1, 2)
    assert torch.equal(tensor, convert_nhwc_to_nchw(result))
    return result

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger



def sample_sgms(model, x_orig, y_orig, n_samples=100):
    """
    performs n times a forward pass and saves the returned segmentations in a list (the accumulated outputs)
    params:
        model: torch.nn.module
        x_orig: torch Tensor with shape (n_batch,n_channels,d1,d2). The input data to the model
        y_orig: torch Tensor with same shape as x_orig. The segmentation as label
    returns: torch tensor with shape (n_samples, n_channels, n_categories, d_1, d_2)
    """

    model.eval() # toggle to evaluation mode
    if torch.cuda.is_available():
        model.cuda()
        x_orig = x_orig.cuda()
        y_orig = y_orig.cuda()
    
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_orig, y_orig, training=False)
            accumulated = model.accumulate_output(out)
            samples.append(accumulated)
    return torch.stack(samples)



def ce_error_map_from_sgms(sgm_samples):
    """
    computes cross entropy error maps between the individual samples and the mean of the samples
    params:
        sgm_samples: torch tensor with shape (n_samples, n_channels, n_categoties, d1, d2). The output of sample_sgms.
    returns: numpy array with shape (n_samples, n_channels, d1, d2). Contains the cross entropy loss between every sample and the mean of samples.
    """

    # shape of mean: (n_channels, d1, d2)
    # compute one-hot encoding of mean prediction
    mean_sgm = torch.argmax(torch.mean(sgm_samples, axis=0), dim=1)
    ce_errors = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for i in range(len(sgm_samples)):
            ce_loss = loss_fn(sgm_samples[i], mean_sgm)
            # print(ce_loss.shape)
            ce_errors.append(ce_loss.cpu().numpy())
    return np.array(ce_errors)

def plot_mean_ce_error_map(ce_errors):
    """
    plots the mean cross entropy map (or expected cross entropy loss)
    params: 
        ce_errors: numpy array with shape(n_errors, n_channels, d1, d2)
    """
    d1, d2 = ce_errors.shape[2], ce_errors.shape[3]
    mean_error_map = np.mean(ce_errors, axis=0).reshape((d1, d2))
    plt.imshow(mean_error_map)
    plt.show()

def ce_error_map_gt_samples(seg_samples, seg_gt):
    """
    computes the error maps between the samples and ground truth annotations 
    params:
        seg_samples: torch tensor with shape (m_samples, n_channels, n_categories, d1, d2). Samples obtained from sample_sgms
        seg_gt: torch tensor with shape (n_annotations, n_channels, d1, d2). The ground-truth annotations
    returns: numpy array with shape (m_samples*n_annotations, n_channels, d1, d2). Contains the cross entropy losses between ground truth and samples.
    """

    ce_errors = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    M = seg_samples.shape[0]
    N = seg_gt.shape[0]
    print('samples shape', seg_samples.shape)
    print('gt shape', seg_gt.shape)
    with torch.no_grad():
        for i in range(M):
            for j in range(N):
                ce_loss = loss_fn(seg_samples[i].cpu(), seg_gt[j].cpu())
                ce_errors.append(ce_loss.cpu().numpy())
    ce_errors = np.array(ce_errors)
    return ce_errors

def ce_error_map_gts(gt_annot):
    """
    computes error map between mean_gt annotation and individual segmentations
    params:
        gt_annot: torch tensor in shape (n_annotations, n_channels, d1, d2)
    returns: numpy array with shape (n_annotations, n_channels, d1, d2). Contains the cross entropy errors between mean gt annotation and individual gt annotations
    """

    # define other CE loss:
    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray        
        Returns: scalar
        """
        ce = -targets*np.log(predictions+1e-9)
        return ce
    
    gt_annot = gt_annot.cpu().numpy()
    mean_annot = np.mean(gt_annot, axis=0)

    ce_errors = []
    for i in range(len(gt_annot)):
        ce_loss = cross_entropy(mean_annot, gt_annot[i])
        ce_errors.append(ce_loss)
    return np.array(ce_errors)

def produce_out_list(model, loader):
    """
    produces a list of model forward passes. 
    params:
        model: torch module; phiseg model
        loader: pytorch data_loader for LIDC dataset
    returns a list of model outputs in shape (n_samples, latent_levels, n_channels, n_categories, d1, d2)
    and a list of labels with shape (n_samples, n_batch, n:channels, d1, d2)
    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    out_list = []
    y_list = []
    with torch.no_grad():
        for x,y,_,_ in loader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x,y)
            out_list.append(out)
            y_list.append(y.cpu().numpy())
            del(x)
            del(y)
            del(out)
    return out_list, y_list

def acc_latents(segms):
    """
    Accumulates the output of the model for each latent level.
    params: 
        segms: torch Tensor with shape (latent_levels, n_channels, n_categories, d1, d2). The model output of one forward pass
    returns the accumulated output for each latent level with same shape as input
    """

    latent_lvls = len(segms)
    latents_acc = [None] * latent_lvls
    latents_acc[latent_lvls-1] = segms[latent_lvls-1]
    for i in reversed(range(latent_lvls-1)):
        latents_acc[i] = latents_acc[i+1] + segms[i]
    return latents_acc

def apply_softmax(accums):
    """
    Applies softmax for accumulated output at each latent level and select the predicted category.
    params:
        accums: torch tensor with shape (latent_levels, n_channels, n_categories, d1, d2)
    returns: numpy array with shape (latent_levels, n_channels, d1, d2)
    """

    latent_lvls = len(accums)
    soft_list = [None] * latent_lvls
    for i in range(latent_lvls):
        soft_list[i] = np.argmax(torch.nn.functional.softmax(accums[i], dim=1).cpu().numpy(), axis=1)
    return np.array(soft_list)

# now all together
def plot_original_and_latent_lvls(orig_label, model_output):
    """
    plots the accumulated model output at each latent level and corresponding real label(s)
    (maybe adapt in future for to display all labels)
    params:
        orig_label: numpy array with shape (n_channels, d1, d2). One of the original annotations.
        model_output: torch tensor with shape (latent_levels, n_channels, n_categories, d1, d2)
    """

    # first: accumulate outputs
    accumulated = acc_latents(model_output)

    # apply softmax and select category 
    softs = apply_softmax(accumulated)

    # define plot with two rows and 5 columns; in top row: original, in bottom row: the latent level softmax stuff
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    gs = GridSpec(2, 5, figure=fig)

    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[1, 4])

    ax1.imshow(orig_label.reshape((128,128)))

    ax2.imshow(softs[0].reshape((128,128)))
    ax3.imshow(softs[1].reshape((128,128)))
    ax4.imshow(softs[2].reshape((128,128)))
    ax5.imshow(softs[3].reshape((128,128)))
    ax6.imshow(softs[4].reshape((128,128)))

    plt.show()

def plot_gt_segmentations(gt_segmentations):
    """
    plots GT annotations from dataset
    params:
        gt_segmentations: torch tensor with shape (n_segmentations, n_channels, d1, d2)
    """
    d1, d2 = gt_segmentations.shape[2], gt_segmentations.shape[3]
    fig, ax = plt.subplots(1,4, figsize=(15,7))
    for i in range(len(gt_segmentations)):
        ax[i].imshow(gt_segmentations[i].detach().cpu().numpy().reshape((d1,d2)))
    plt.show()

# def eval_ged(model, loader, n_samples=50):
#     """
#     computes the mean generalized energy distance between samples and original labels for validation
#     params:
#         model: torch module. The PHiSeg model
#         loader: Dataloader for the LIDC dataset
#         n_samples: int. How many samples to draw for GED evaluation
#     """
#     model.eval()
#     ged_list = []
#     with torch.no_grad():
#         for x, y, _, labels in loader:
#             # sample 
#             samples = sample_sgms(model, x, y, n_samples=n_samples)
#             samples = np.asarray(samples.cpu())
#             samples_argmax = np.argmax(samples, axis=2).reshape((n_samples, 128,128))

#             # compute GED
#             labels_npy = np.asarray(labels.cpu()).reshape((4,128,128))
#             ged = generalised_energy_distance(samples_argmax, labels_npy, 2)
#             ged_list.append(ged)
#             # print(ged)

#             # delete some stuff:
#             del(samples)
#             del(samples_argmax)
#             del(labels)
#             del(ged)
#     ged_arr = np.asarray(ged_list)
#     return np.mean(ged_arr)


# def generate_n_samples(model, x, n_samples=100):
#     """
#     performs n times a forward pass and saves the returned segmentations in a list (the accumulated outputs)
#     params:
#         model: torch.nn.module
#         x_orig: torch Tensor with shape (n_batch,n_channels,d1,d2). The input data to the model
#     returns: torch tensor with shape (n_samples, n_channels, n_categories, d_1, d_2)
#     """
#     if torch.cuda.is_available():
#         model.cuda()
#         x = x.cuda()
#     bs = x.shape[0]
#     xb = torch.vstack([x for _ in range(n_samples)])
#     with torch.no_grad():
#         # encode
#         prior_latent_space, _, _ = model.prior(xb, training_prior=False)
#         # decode
#         s_out_list = model.likelihood(prior_latent_space)
#         accumulated = model.accumulate_output(s_out_list)
#     accumulated = accumulated.view([n_samples, bs] + [*accumulated.shape][1:])
#     return accumulated


# def eval_ged(model, loader, n_samples=50):
#     """
#     computes the mean generalized energy distance between samples and original labels for validation
#     params:
#         model: torch module. The PHiSeg model
#         loader: Dataloader for the LIDC dataset
#         n_samples: int. How many samples to draw for GED evaluation
#     """
#     model.eval()
#     ged_list = []
#     with torch.no_grad():
#         for x, y, _, labels in loader:
#             # sample
#             samples = generate_n_samples(model, x, n_samples=n_samples)
#             samples = np.asarray(samples.cpu())
#             samples_argmax = np.argmax(samples, axis=2).reshape((n_samples, 128,128))

#             # compute GED
#             labels_npy = np.asarray(labels.cpu()).reshape((4,128,128))
#             ged = generalised_energy_distance(samples_argmax, labels_npy, 2)
#             ged_list.append(ged)

#             # delete some stuff:
#             del(samples)
#             del(samples_argmax)
#             del(labels)
#             del(ged)
#     ged_arr = np.asarray(ged_list)
#     return np.mean(ged_arr)

def generate_n_samples(model, x, n):
    elems_at_once = 10
    i_loops = int(n/elems_at_once)
    bs = x.shape[0]
    new_n = int(i_loops*elems_at_once)
    samples = []
    for i in range(i_loops):
        xb = torch.vstack([x for _ in range(elems_at_once)])
        if torch.cuda.is_available():
            model.cuda()
            xb.cuda()
        with torch.no_grad():
            # encode
            prior_latent_space, _, _ = model.prior(xb.cuda(), training_prior=False)
            # decode
            s_out_list = model.likelihood(prior_latent_space)
            accumulated = model.accumulate_output(s_out_list)
        samples.append(accumulated)
    samples = torch.stack(samples).view([new_n, bs] + [*accumulated.shape][1:])
    # print(samples.shape)
    return samples


def pairwise_jaccard_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """
    Input parameters A & B need to be nd.arrays of type 'int' or 'bool'
    Returns a matrix of pairwise jaccard distances of shape [N,M]
    """
    assert A.dtype in ["int", "bool"] and B.dtype in ["int", "bool"]
    assert np.all([int(u) in [0, 1] for u in np.unique(A)])
    assert np.all([int(u) in [0, 1] for u in np.unique(B)])

    B = B.transpose((3, 1, 2, 0))
    intersection = np.sum(A & B, axis=(1, 2))
    union = np.sum(A, axis=(1, 2)) + np.sum(B, axis=(1, 2)) - intersection
    pairwise_jaccard_distances = 1 - (intersection / union)

    # Get rid of the potential nan values again
    pairwise_jaccard_distances[(union == 0) & (intersection == 0)] = 1
    pairwise_jaccard_distances[(union == 0) & (intersection > 0)] = 0

    return pairwise_jaccard_distances


def pairwise_L2_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Returns a matrix of pairwise L2 distances of shape [N,M]"""
    all_differences = A - B.transpose((3, 2, 1, 0))
    pairwise_l2_norms = np.linalg.norm(all_differences, axis=1)
    return pairwise_l2_norms


def generalised_energy_distance(
    x: np.ndarray, y: np.ndarray, metric=pairwise_jaccard_distance
) -> float:
    """
    Calculate the (generalised) energy distance (https://en.wikipedia.org/wiki/Energy_distance)
    where x,y are np.ndarrays containing samples of the distributions to be
    compared for a given metric.
        Parameters:
            x (np.ndarray of shape N x Sx x Sy): One set of N samples
            y (np.ndarray of shape M x Sx x Sy): Another set of M samples
            metric (function): a function implementing the desired metric
        Returns:
            The generalised energy distance of the two samples (float)
    """

    assert x.ndim == 3 and y.ndim == 3

    def expectation_of_difference(a, b):
        N, M = a.shape[0], b.shape[0]
        A = np.tile(a[:, :, :, np.newaxis], (1, 1, 1, M))  # N x Sx x Sy x M
        B = np.tile(b[:, :, :, np.newaxis], (1, 1, 1, N))  # M x Sx x Sy x N
        return metric(A, B).mean()

    Exy = expectation_of_difference(x, y)
    Exx = expectation_of_difference(x, x)
    Eyy = expectation_of_difference(y, y)

    # ed = np.sqrt(2 * Exy - Exx - Eyy)
    ed = 2 * Exy - Exx - Eyy
    return ed**2


def eval_ged(model, loader, n_samples=20):
    """
    computes the mean generalized energy distance between samples and original labels for validation
    params:
        model: torch module. The PHiSeg model
        loader: Dataloader for the LIDC dataset
        n_samples: int. How many samples to draw for GED evaluation
    """
    model.eval()
    ged_list = []
    with torch.no_grad():
        for x, y, _, labels in loader:
            # sample
            samples = generate_n_samples(model, x, n=n_samples)
            samples = np.asarray(samples.cpu())
            samples_argmax = np.argmax(samples, axis=2).reshape((n_samples, 128,128))

            # compute GED
            labels_npy = np.int64(np.asarray(labels.cpu()).reshape((4,128,128)))
            ged = generalised_energy_distance(samples_argmax, labels_npy)
            ged_list.append(ged)

            # delete some stuff:
            del(samples)
            del(samples_argmax)
            del(labels)
            del(ged)
    ged_arr = np.asarray(ged_list)
    return np.mean(ged_arr)


def mask_input(x):
    """
    overlays a hard-coded mask onto a single image or batch 
    params:
        x: torch Tensor with shape (n_batch, n_channels, d1, d2)
    returns: torch Tensor with shape (n_batch, n_channels, d1, d2). The masked image(s)
    """
    mask_template = torch.ones_like(x)
    mask_template[0,0,65:68, 55:70] = 0
    x_masked = x*mask_template
    return x_masked

def dice_coeff(y_gt, y_pred):
    """
    computes the mean dice score of a batch of ground truths and the according predictions
    y_gt: shape (n_batch, n_channels, d1, d2)
    y_pred: shape (n_batch, n_channels, d1, d2)
    """

    eps = 10e-10
    intersection = torch.sum(y_gt*y_pred, axis=(1,2,3))
    sum_gt = torch.sum(y_gt, axis=(1,2,3))
    sum_pred = torch.sum(y_pred, axis=(1,2,3))

    dice_batch = (2* intersection + eps) / (sum_gt + sum_pred + eps)


    return torch.mean(dice_batch)

def dice_coeff_on_loader(model, loader, model_type):
    """
    compute the mean dice coefficient on a evaluation data loader
    params:
        model: torch module. The model to evaluate
        loader: the dataloader
        model_type: string. One of {phiseg, unet}
    returns: average dice coefficient on the validation set
    """

    model.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    dice_coef_list = []
    with torch.no_grad():
        for x,y,_,_ in loader:
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            if model_type=='phiseg':
                n_samples = 20
                samples = generate_n_samples(model, x, n_samples=n_samples)
                # print('samples shape:', samples.shape)
                mean_sample = torch.mean(samples, axis=0)
                mean_samples_argmax = torch.argmax(mean_sample, axis=1).reshape((1,1, 128,128))
                # print(mean_samples_argmax.shape)
                dice = dice_coeff(mean_samples_argmax, y)
                # print('dice:',dice)
                dice_coef_list.append(dice.cpu().numpy())
            else:
                output = model(x.cuda())
                argmax_out = torch.argmax(output, dim=1).reshape((1,1,128,128))
                dice = dice_coeff(argmax_out, y)
                # print('dice:', dice)
                dice_coef_list.append(dice.cpu().numpy())
    return np.mean(np.array(dice_coef_list))