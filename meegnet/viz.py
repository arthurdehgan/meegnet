import os
import mne
import logging
import copy
import torch
from mne.viz import plot_topomap
from torch.autograd import Variable
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from matplotlib.gridspec import GridSpec
from collections.abc import Iterable
from joblib import Parallel, delayed
from meegnet.utils import compute_psd, cuda_check
from pytorch_grad_cam import GuidedBackpropReLUModel

LOG = logging.getLogger("meegnet")
mne.set_log_level(False)


def compute_saliency_maps(
    dataset,
    labels,
    sub,
    sal_path,
    net,
    threshold,
    epoched=False,
):

    device = cuda_check()
    GBP = GuidedBackpropReLUModel(net, device=device)

    # Load all trials and corresponding labels for a specific subject.
    data = dataset.data
    targets = dataset.labels
    if epoched:
        target_saliencies = [[[], []], [[], []]]
    else:
        target_saliencies = [[], []]

    # For each of those trial with associated label:
    for trial, label in zip(data, targets):
        X = trial
        while len(X.shape) < 4:
            X = X[np.newaxis, :]
        X = X.to(torch.float64).to(device)
        # Compute predictions of the trained network, and confidence
        preds = torch.nn.Softmax(dim=1)(net(X)).detach().cpu()
        pred = preds.argmax().item()
        confidence = preds.max()
        label = int(label)

        # If the confidence reaches desired treshhold (given by args.confidence)
        if confidence >= threshold and pred == label:
            # Compute Guided Back-propagation for given label projected on given data X
            guided_grads = GBP(X.to(device), label)
            guided_grads = np.rollaxis(guided_grads, 2, 0)
            # Compute saliencies
            pos_saliency, neg_saliency = get_positive_negative_saliency(guided_grads)

            # Depending on the task, add saliencies in lists
            if epoched:
                target_saliencies[label][0].append(pos_saliency)
                target_saliencies[label][1].append(neg_saliency)
            else:
                target_saliencies[0].append(pos_saliency)
                target_saliencies[1].append(neg_saliency)
    # With all saliencies computed, we save them in the specified save-path
    n_saliencies = 0
    n_saliencies += sum([len(e) for e in target_saliencies[0]])
    n_saliencies += sum([len(e) for e in target_saliencies[1]])
    LOG.info(f"{n_saliencies} saliency maps computed for {sub}")
    for j, sal_type in enumerate(("pos", "neg")):
        if epoched:
            for i, label in enumerate(labels):
                sal_filepath = os.path.join(
                    sal_path,
                    f"{sub}_{labels[i]}_{sal_type}_sal_{threshold}confidence.npy",
                )
                np.save(sal_filepath, np.array(target_saliencies[i][j]))
        else:
            lab = "" if not epoched else f"_{labels[label]}"
            sal_filepath = os.path.join(
                sal_path,
                f"{sub}{lab}_{sal_type}_sal_{threshold}confidence.npy",
            )
            np.save(sal_filepath, np.array(target_saliencies[j]))


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists("../results"):
        os.makedirs("../results")
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    # path_to_file = os.path.join("../results", file_name + ".jpg")
    path_to_file = os.path.join(file_name + ".jpg")
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists("../results"):
        os.makedirs("../results")
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, "hsv")
    # Save colored heatmap
    path_to_file = os.path.join("../results", file_name + "_Cam_Heatmap.png")
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join("../results", file_name + "_Cam_On_Image.png")
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join("../results", file_name + "_Cam_Grayscale.png")
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def choose_best_window(data, fs=500, w_size=300):
    """
    data: array
        Must be of size k x n_samples. k can be sensor dimension or trial dimension.
    w_size: int
        The size of the window in ms
    """
    # Generate masks over time samples where sample is superior to 2x std of the vector
    masks = [dat >= (np.mean(dat) + np.std(dat) * 2) for dat in data]
    w_size = int(w_size * fs / 1000)
    best_window_idx = Parallel(n_jobs=-1)(delayed(best_window)(mask, w_size) for mask in masks)
    return best_window_idx


def best_window(mask, w_size):
    # create an array of all the possible windows in the trial with their mask values
    windows = np.array([mask[i : i + w_size] for i in range(len(mask) - w_size)])
    # get the sum of mask values in each windows. High number indicates lots of True values
    values = [sum(window) for window in windows]
    if max(values) == 0:
        return None
    # best window is where the values are the highest
    best_window_index = np.where(values == max(values))[0]
    # below is to handle if there are multiple best windows with the same values, in which case we
    # try to check if they overlap, and keep only if they are separated.
    if len(best_window_index) > 1:
        idx_range = best_window_index[-1] - best_window_index[0]
        # duration betweeen first and last window must be w_size/2
        if idx_range <= int(w_size / 2):
            # Then best would be in the middle of all this
            best = int(len(best_window_index) / 2)
            return best_window_index[best]
        elif idx_range <= w_size:
            return (best_window_index[0], best_window_index[-1])
        else:
            best = int(len(best_window_index) / 2)
            return (best_window_index[0], best, best_window_index[-1])
    else:
        # only one window, just add it.
        return best_window_index[0]


def compute_single_sal_psd(index, trial, w_size, fs):
    if isinstance(index, Iterable):
        tmp = []
        for idx in index:
            window = trial[idx : idx + w_size]
            while len(window.shape) < 3:
                window = window[np.newaxis, :]
            tmp.append(compute_psd(window, fs=fs))
        return np.mean(tmp, axis=0)
    else:
        if index is not None:
            window = trial[index : index + w_size]
            while len(window.shape) < 3:
                window = window[np.newaxis, :]
            return compute_psd(window, fs=fs)
        else:
            return [None] * 7


def compute_saliency_based_psd(saliency, trial, w_size, fs):
    chan_data = []
    for i, chan in enumerate(saliency):
        windows_idx = choose_best_window(chan, fs, w_size)
        transformed_data = Parallel(n_jobs=1)(
            delayed(compute_single_sal_psd)(index, trial[i, j], w_size, fs)
            for j, index in enumerate(windows_idx)
        )
        chan_data.append(np.array(transformed_data).squeeze())
    return np.array(chan_data)


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (
        ("../input_images/snake.jpg", 56),
        ("../input_images/cat_dog.png", 243),
        ("../input_images/spider.png", 72),
    )
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind("/") + 1 : img_path.rfind(".")]
    # Read image
    original_image = Image.open(img_path).convert("RGB")
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (
        original_image,
        prep_img,
        target_class,
        file_name_to_export,
        pretrained_model,
    )


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists("../results"):
        os.makedirs("../results")
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    # path_to_file = os.path.join("../results", file_name + ".jpg")
    path_to_file = os.path.join(file_name + ".jpg")
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists("../results"):
        os.makedirs("../results")
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, "hsv")
    # Save colored heatmap
    path_to_file = os.path.join("../results", file_name + "_Cam_Heatmap.png")
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join("../results", file_name + "_Cam_On_Image.png")
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join("../results", file_name + "_Cam_Grayscale.png")
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def choose_best_window(data, fs=500, w_size=300):
    """
    data: array
        Must be of size k x n_samples. k can be sensor dimension or trial dimension.
    w_size: int
        The size of the window in ms
    """
    # Generate masks over time samples where sample is superior to 2x std of the vector
    masks = [dat >= (np.mean(dat) + np.std(dat) * 2) for dat in data]
    w_size = int(w_size * fs / 1000)
    best_window_idx = Parallel(n_jobs=-1)(delayed(best_window)(mask, w_size) for mask in masks)
    return best_window_idx


def best_window(mask, w_size):
    # create an array of all the possible windows in the trial with their mask values
    windows = np.array([mask[i : i + w_size] for i in range(len(mask) - w_size)])
    # get the sum of mask values in each windows. High number indicates lots of True values
    values = [sum(window) for window in windows]
    if max(values) == 0:
        return None
    # best window is where the values are the highest
    best_window_index = np.where(values == max(values))[0]
    # below is to handle if there are multiple best windows with the same values, in which case we
    # try to check if they overlap, and keep only if they are separated.
    if len(best_window_index) > 1:
        idx_range = best_window_index[-1] - best_window_index[0]
        # duration betweeen first and last window must be w_size/2
        if idx_range <= int(w_size / 2):
            # Then best would be in the middle of all this
            best = int(len(best_window_index) / 2)
            return best_window_index[best]
        elif idx_range <= w_size:
            return (best_window_index[0], best_window_index[-1])
        else:
            best = int(len(best_window_index) / 2)
            return (best_window_index[0], best, best_window_index[-1])
    else:
        # only one window, just add it.
        return best_window_index[0]


def compute_single_sal_psd(index, trial, w_size, fs):
    if isinstance(index, Iterable):
        tmp = []
        for idx in index:
            window = trial[idx : idx + w_size]
            while len(window.shape) < 3:
                window = window[np.newaxis, :]
            tmp.append(compute_psd(window, fs=fs))
        return np.mean(tmp, axis=0)
    else:
        if index is not None:
            window = trial[index : index + w_size]
            while len(window.shape) < 3:
                window = window[np.newaxis, :]
            return compute_psd(window, fs=fs)
        else:
            return [None] * 7


def compute_saliency_based_psd(saliency, trial, w_size, fs):
    chan_data = []
    for i, chan in enumerate(saliency):
        windows_idx = choose_best_window(chan, fs, w_size)
        transformed_data = Parallel(n_jobs=1)(
            delayed(compute_single_sal_psd)(index, trial[i, j], w_size, fs)
            for j, index in enumerate(windows_idx)
        )
        chan_data.append(np.array(transformed_data).squeeze())
    return np.array(chan_data)


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (
        ("../input_images/snake.jpg", 56),
        ("../input_images/cat_dog.png", 243),
        ("../input_images/spider.png", 72),
    )
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind("/") + 1 : img_path.rfind(".")]
    # Read image
    original_image = Image.open(img_path).convert("RGB")
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (
        original_image,
        prep_img,
        target_class,
        file_name_to_export,
        pretrained_model,
    )


def make_gif(image_list, output=None, duration=100, loop=0):
    if output is None:
        output = ".".join(image_list[0].split(".")[:-1] + ["gif"])
    frames = [Image.open(image) for image in image_list]
    frame_one = frames[0]
    frame_one.save(
        output,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
    )


def avg_range(arr: list):
    length = int(len(arr) / 2)
    return rec_avg_range(list(arr), length)


def rec_avg_range(arr: list, length: int):
    if len(arr) <= 1:
        return 0
    argmax = np.argmax(arr)
    max = arr.pop(argmax)
    argmin = np.argmin(arr)
    min = arr.pop(argmin)
    range = (max - min) / length
    return range + rec_avg_range(arr, length)


def _unit_test_avg_range():
    liste = [18, -8, -6, 3, -10, 15, 0]
    assert avg_range(liste) == 20


def generate_saliency_figure(
    saliencies: dict,
    info,
    save_path: str = "",
    suffix: str = "",
    title: str = "",
    sensors: list = [""],
    sfreq=500,
    edge=100,
    cmap="coolwarm",
    stim_tick=None,
    show=False,
    outlines=None,
    topomap="window",
):
    """
    Generates a figure visualizing saliency maps for MEG data.

    This function creates a grid of images showing the saliency maps for
    types of stimuli (e.g., image and sound) and sensor channels (e.g., MAG, GRAD1, GRAD2).
    It also plots a topomap for the maximum saliency index along with a color bar.

    Parameters
    ----------
    saliencies : dict
        Dictionary containing saliency maps. Keys should correspond to different types
        of stimuli (e.g., "image", "sound"), and values should be numpy arrays of shape
        3 x sensors x samples, representing the saliency maps for each channel.
    save_path : str, optional
        Path to save the generated figure. Default is an empty string, which means the
        figure will not be saved automatically.
    suffix : str, optional
        Suffix to append to the filename when saving the figure. Default is an empty string.
    title : str, optional
        Title for the figure. Default is an empty string.
    sensors : list, optional
        List of sensor types to include in the visualization. Default is [""].
    sfreq : int, optional
        Sampling frequency for computation of  xticks. Default is  500.
    cmap : str, optional
        Colormap to use for displaying the topo- and saliency maps. Default is "coolwarm".
    stim_tick : int, optional
        Tick position for the stimulus event in the time axis. Default is None,
    show : bool, optional
        Wether to show the figure or not. Useful for ipynb.
    topomap : str, optional
        Must be "window", "average" or "timing". if " average, the average saliency across time is used. If
        timing, then the saliency at to the highest saliency timing is used. if "window" is used,
        then the saliency window around the max saliency timing is used.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that the input saliency maps are normalized to have zero mean and
    unit variance. The colormap used for displaying the saliency maps is "coolwarm" by default,
    but can be changed with the `cmap` parameter.

    The function creates a grid layout with a subplot for each sensor channel and a subplot for the
    topomap. The grid layout is dynamically adjusted based on the number of sensors.

    The function does not handle exceptions that may occur during the plotting process, such as issues with
    file I/O or invalid input data.
    """

    assert topomap in (
        "timing",
        "average",
        "window",
    ), f"{topomap} is not a valid option for the topomap parameter. ('timing', 'average', 'window')"
    tick_ratio = 1000 / sfreq
    padding = int(edge / tick_ratio)
    if stim_tick is not None:
        stim_tick -= edge
    if suffix != "" and not suffix.endswith("_"):
        suffix += "_"
    n_blocs = len(sensors)  # number of blocs of figures in a line
    n_lines = len(saliencies)  # number of lines for the pyplot figure
    n_cols = n_blocs * 3 + 1  # number of columns for the pyplot figure
    grid = GridSpec(n_lines, n_cols)
    fig = plt.figure(figsize=(n_cols * 2, n_lines * 2))
    plt.title(title)
    plt.axis("off")
    axes = []
    # First pass to gather vlim values:
    vlim = 0
    for i, label in enumerate(saliencies.keys()):
        gradient = copy.copy(saliencies[label].squeeze())
        gradient /= np.abs(gradient).max()
        for j, sensor_type in zip(range(0, n_blocs * 3, n_blocs), sensors):
            idx = j // 3
            grads = gradient[idx][:, padding:-padding]
            segment_length = grads.shape[1]
            # mid_slice = (0, segment_length)
            # gradmeans = grads[:, mid_slice[0] : mid_slice[1]].mean(axis=1)[:, np.newaxis]
            # grads -= gradmeans  # We remove mean accross time to make the variations accross time pop-up more
            vmax = grads.max()
            vmin = grads.min()
            vlim_curr = max(abs(vmax), abs(vmin))
            if vlim_curr > vlim:
                vlim = vlim_curr

    for i, label in enumerate(saliencies.keys()):
        gradient = saliencies[label].squeeze()
        assert (
            len(gradient) == n_blocs
        ), "Can't generate figures for all sensors, check if the saliencies have been properly computed."
        gradient /= np.abs(gradient).max()
        for j, sensor_type in zip(range(0, n_blocs * 3, n_blocs), sensors):
            idx = j // 3
            # grads = gradient[idx]
            # In an attempt to remove the edge effect:
            # We remove the first and last edge points -> therefore tick is moved to 25 (was 75)
            grads = gradient[idx][:, padding:-padding]

            segment_length = grads.shape[1]
            # We add the mid_slice variable in an attempt to tackle the edge effects by removing mean from center values for example
            # But it was uneffective. This could still be useful so we leave it here...
            # mid_slice = (0, segment_length)
            # mid_slice = (int(segment_length / 4), int(3 * segment_length / 4))
            # gradmeans = grads[:, mid_slice[0] : mid_slice[1]].mean(axis=1)[:, np.newaxis]
            # grads -= gradmeans  # We remove mean accross time to make the variations accross time pop-up more
            n_sensors = grads.shape[0]
            max_idx = np.unravel_index(abs(grads).argmax(), grads.shape)[1]
            # max_idx = np.argmax([avg_range(arr) for arr in grads.T])
            # max_idx = np.argmax(np.mean(grads, axis=0))
            axes.append(fig.add_subplot(grid[i, j : j + 2]))
            plt.imshow(
                grads,
                interpolation="nearest",
                aspect=1,
                vmin=-vlim,
                vmax=vlim,
                cmap=cmap,
            )
            axes[-1].spines["top"].set_visible(False)
            axes[-1].spines["right"].set_visible(False)
            axes[-1].yaxis.tick_right()

            if stim_tick is not None:
                plt.axvline(x=stim_tick, color="black", linestyle="--", linewidth=1)

            stim_tick_index = 0 if stim_tick is None else stim_tick
            x_ticks = [0, stim_tick_index, int(segment_length / tick_ratio), segment_length]
            if topomap != "average":
                plt.axvline(x=max_idx, color="green", linestyle="--", linewidth=1)
                x_ticks += [max_idx]
            x_ticks = sorted(x_ticks)
            ticks_values = [
                (x_tick - stim_tick_index) * tick_ratio + edge for x_tick in x_ticks
            ]
            plt.xticks(x_ticks, ticks_values, fontsize=8)
            plt.yticks([0, n_sensors], [n_sensors, 0])

            if j == 0:
                axes[-1].text(-50, 50, label, ha="left", va="center", rotation="vertical")
            if idx == n_blocs - 1:
                axes[-1].yaxis.set_label_position("right")
                plt.ylabel("sensors")
            if i == 0:
                plt.title(sensor_type)
            if i == 1:
                plt.xlabel("time (ms)")
            axes.append(fig.add_subplot(grid[i, j + 2]))
            if topomap == "timing":
                data = grads[:, max_idx]
            elif topomap == "window":
                start = max_idx - int(segment_length / 8)
                end = max_idx + int(segment_length / 8)
                if start < 0:
                    start = 0
                    end = start + int(segment_length / 4)
                elif end > segment_length:
                    end = segment_length
                    start = 3 * int(segment_length / 4)
                data = grads[:, start:end].mean(axis=1)
            else:
                data = grads.mean(axis=1)

            im, _ = plot_topomap(
                data.ravel(),
                info,
                res=300,
                cmap=cmap,
                vlim=(-vlim, vlim),
                show=False,
                contours=0,
                axes=axes[-1],
                outlines=outlines,
            )
            if idx == n_blocs - 1:
                axes.append(fig.add_subplot(grid[i, n_blocs * 3]))
                fig.colorbar(
                    im,
                    ax=axes[-1],
                    location="right",
                    shrink=0.9,
                    ticks=(-vlim, 0, vlim),
                )
                axes[-1].axis("off")

    out_path = os.path.join(save_path, f"{suffix}saliencies.png")
    # plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()
    return out_path


def colorFader(c1, c2, mix=0, alpha=0.5):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def plot_masked_epoch(data, mask, c1="white", c2="green", alpha=0.3, title: str = None):
    dim = len(data.shape)
    assert data.shape == mask.shape, "Shape of the data should match shape of the mask."
    assert (
        dim == 2
    ), f"Data is supposed to be a 2 dimension array, a {dim}-dimension array was given."

    mask = (mask - mask.min()) / np.ptp(mask)
    fig, axes = plot_epoch(data, title)
    for i, ax in enumerate(axes):
        for j, x in enumerate(range(data.shape[1])):
            ax.axvline(
                x + 1,
                color=colorFader(c1, c2, mask[i, j]),
                zorder=0,
                alpha=alpha,
                linewidth=50,
            )

    return fig, axes


def plot_epoch(data, title: str = None):
    dim = len(data.shape)
    assert (
        dim == 2
    ), f"Data is supposed to be a 2 dimension array, a {dim}-dimension array was given."

    grid = GridSpec(len(data), 1, hspace=0)
    fig = plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.axis("off")
    axes = []
    for i, chan in enumerate(data):
        ax = fig.add_subplot(grid[i, 0])
        plt.plot(chan, color="black", linewidth=1)
        plt.xlim(0, data.shape[1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # plt.ylabel(i, rotation="horizontal")
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        if i != len(data) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        axes.append(ax)
    plt.tight_layout()

    # out_path = os.path.join(save_path, "visualizations", "average_trial.png")
    # plt.savefig(out_path, dpi=300)
    # return out_path
    return fig, axes
