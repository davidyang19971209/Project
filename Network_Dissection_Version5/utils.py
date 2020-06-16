from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from zca import ZCA


def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    nimg = nimg.get_fdata()
    img_list = []
    #affine 从image data array到标准参考空间的映射
    #header记录属性信息，采集设备名称，体素大小，扫描层数
    z = nimg.shape[2]
    for i in range (60,120,2):
        img_list.append(nimg[:,:,i])
    # img_list.append(nimg[:,:,66])
    return img_list

# def dsc(y_pred, y_true, lcc=True):
#     if lcc and np.any(y_pred):
#         # y_pred = np.round(y_pred).astype(int)
#         y_true = np.round(y_true).astype(int)
#         # y_pred = largest_connected_component(y_pred)
#     return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def dsc(y_pred, y_true, lcc=True):
    if lcc and np.any(y_pred):
        max_pred=np.max(y_pred,axis=1)
        
        b,h,w = max_pred.shape
        max_pred = max_pred.reshape(b,1,h,w)
        
        one_hot = (y_pred == max_pred)+0
        
        loss=0
        loss_list = []
        
        for i in range (one_hot.shape[1]):
            temp=single_dsc(one_hot[:,i,:,:],y_true[:,i,:,:])
            loss+=temp
            loss_list.append(temp)
    return loss/one_hot.shape[1],loss_list;
        # skip the batch and class axis for calculating Dice score
    #     axes = tuple(range(2, len(one_hot.shape))) 
    #     numerator = 2. * np.sum(one_hot * y_true, axes)
    #     denominator = np.sum(one_hot + y_true, axes)
        
    #     embed()
    # return (numerator / denominator).mean()

def single_dsc(y_pred,y_true):
 
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + 1) / (y_pred.sum() + y_true.sum() + 1)
    
    return dsc

def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]

    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))

    padding = padding + ((0, 0),)
    mask = np.pad(mask, padding, mode="edge")
    volume = np.pad(volume, padding, mode="edge")
    return volume, mask


def resize_sample(x, size=240):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)

    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        anti_aliasing=False,
    )
    print("finish")
    return volume.round(0), mask


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


def log_images(x, y_true, y_pred, channel=1):
    images = []
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(x_np.shape[0]):
        image = gray2rgb(np.squeeze(x_np[i]))
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

def make_one_hot(mask):
    class_1 = (mask[:,:,:,0]==0)+0
    class_2 = (mask[:,:,:,0]==1)+0
    class_3 = (mask[:,:,:,0]==2)+0
    class_4 = (mask[:,:,:,0]==4)+0
    
    # class_2 = class_2+class_4

    b,h,w = class_1.shape

    class_1 = class_1.reshape(b,h,w,1)
    class_2 = class_2.reshape(b,h,w,1)
    class_3 = class_3.reshape(b,h,w,1)
    class_4 = class_4.reshape(b,h,w,1)

    result = np.concatenate((class_1,class_2,class_3,class_4),axis=3)
    # result = np.concatenate((class_1,class_2,class_3),axis=3)
    return result

# def make_one_hot(mask):
#     # class_1 = (mask[:,:,:,0]==0)+0
#     class_2 = (mask[:,:,:,0]==1)+0
#     class_3 = (mask[:,:,:,0]==2)+0
#     # class_4 = (mask[:,:,:,0]==4)+0

#     b,h,w = class_2.shape

#     # class_1 = class_1.reshape(b,h,w,1)
#     class_2 = class_2.reshape(b,h,w,1)
#     class_3 = class_3.reshape(b,h,w,1)
#     # class_4 = class_4.reshape(b,h,w,1)

#     # result = np.concatenate((class_1,class_2,class_3,class_4),axis=3)
#     result = np.concatenate((class_2,class_3),axis=3)
#     return result

def convert_mask_to_one(mask):
    class_1 = (mask[:,:,:,0]== np.max(mask[:,:,:,0]))+0
    class_2 = (mask[:,:,:,1]!= np.min(mask[:,:,:,1]))+0
    class_3 = (mask[:,:,:,2]!= np.min(mask[:,:,:,2]))+0
    class_4 = (mask[:,:,:,3]!= np.min(mask[:,:,:,3]))+0

    b,h,w = class_1.shape

    class_1 = class_1.reshape(b,h,w,1)
    class_2 = class_2.reshape(b,h,w,1)
    class_3 = class_3.reshape(b,h,w,1)
    class_4 = class_4.reshape(b,h,w,1)

    result = np.concatenate((class_1,class_2,class_3,class_4),axis=3)
    # result = np.concatenate((class_1,class_2,class_3),axis=3)
    return result

# def convert_mask_to_one(mask):
#     class_1 = (mask[:,:,:,0]!= np.min(mask[:,:,:,0]))+0
#     class_2 = (mask[:,:,:,1]!= np.min(mask[:,:,:,1]))+0
#     # class_3 = (mask[:,:,:,2]!= np.min(mask[:,:,:,2]))+0
#     # class_4 = (mask[:,:,:,3]!= np.min(mask[:,:,:,3]))+0

#     b,h,w = class_1.shape

#     class_1 = class_1.reshape(b,h,w,1)
#     class_2 = class_2.reshape(b,h,w,1)
#     # class_3 = class_3.reshape(b,h,w,1)
#     # class_4 = class_4.reshape(b,h,w,1)

#     # result = np.concatenate((class_1,class_2,class_3,class_4),axis=3)
#     result = np.concatenate((class_1,class_2),axis=3)
#     return result

def zca_whitening(image):
    data = image
    zca = ZCA()
    for i in range(data.shape[0]):
        data[i,:,:,0] = zca.fit_transform(data[i,:,:,0])
    return data
