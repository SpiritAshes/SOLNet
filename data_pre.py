import os
import numpy as np
import shutil
import os
from PIL import Image
import random
from PIL import ImageEnhance

#several data augumentation strategies
def cv_random_left_right_flip(img, gt):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, gt

def cv_random_top_bottom_flip(img, gt):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
    return img, gt

def randomCrop(image, gt):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), gt.crop(random_region)

def randomRotation(image, gt, random_angle):
    mode=Image.BICUBIC
    # random_angle = np.random.randint(-15, 15)
    image=image.rotate(random_angle, mode)
    gt=gt.rotate(random_angle, mode)
    return image, gt

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img) 

def filter_files(images, gts):
    assert len(images) == len(gts)
    images = []
    # depths = []
    gts = []
    for img_path, gt_path in zip(images, gts):
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        if img.size == gt.size:
            images.append(img_path)
            gts.append(gt_path)
    images = images
    gts = gts

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def data_aug(image_path, datasets, save_path):

    save_aug_images_path = image_path + save_path + '/train/images/'
    save_aug_gts_path = image_path + save_path + '/train/GT/'
    if not os.path.exists(save_aug_images_path):
        os.makedirs(save_aug_images_path)
    if not os.path.exists(save_aug_gts_path):
        os.makedirs(save_aug_gts_path)

    for dataset in datasets:
        image_root = image_path + dataset + '/train/images/'
        gt_root = image_path + dataset + '/train/GT/'
        images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        images = sorted(images)
        gts = sorted(gts)
        filter_files(images, gts)

        for i, image in enumerate(images):
            image = rgb_loader(images[i])
            gt = binary_loader(gts[i])
            image_Enhance = colorEnhance(image)
            gt_Enhance = gt
            image_Crop, gt_Crop = randomCrop(image, gt)
            image_Gaussian = randomGaussian(image)
            gt_Gaussian = gt

            image_Rotation_90, gt_Rotation_90 = randomRotation(image, gt, 90)
            image_Crop_90, gt_Crop_90 = randomCrop(image_Rotation_90, gt_Rotation_90)

            image_Rotation_180, gt_Rotation_180 = randomRotation(image, gt, 180)
            image_Crop_180, gt_Crop_180 = randomCrop(image_Rotation_180, gt_Rotation_180)

            image_Rotation_270, gt_Rotation_270 = randomRotation(image, gt, 270)
            image_Crop_270, gt_Crop_270 = randomCrop(image_Rotation_270, gt_Rotation_270)

            left_right_flip_image, left_right_flip_gt = cv_random_left_right_flip(image, gt)
            left_right_flip_image_Crop, left_right_flip_gt_Crop = randomCrop(left_right_flip_image, left_right_flip_gt)
            left_right_flip_image_Enhance = colorEnhance(left_right_flip_image)
            left_right_flip_gt_Enhance = left_right_flip_gt
            left_right_flip_image_Gaussian = randomGaussian(left_right_flip_image)
            left_right_flip_gt_Gaussian = left_right_flip_gt

            left_right_flip_image_Rotation_90, left_right_flip_gt_Rotation_90 = randomRotation(left_right_flip_image, left_right_flip_gt, 90)
            left_right_flip_image_Crop_90, left_right_flip_gt_Crop_90 = randomCrop(left_right_flip_image_Rotation_90, left_right_flip_gt_Rotation_90)

            left_right_flip_image_Rotation_180, left_right_flip_gt_Rotation_180 = randomRotation(left_right_flip_image, left_right_flip_gt, 180)
            left_right_flip_image_Crop_180, left_right_flip_gt_Crop_180 = randomCrop(left_right_flip_image_Rotation_180, left_right_flip_gt_Rotation_180)

            left_right_flip_image_Rotation_270, left_right_flip_gt_Rotation_270 = randomRotation(left_right_flip_image, left_right_flip_gt, 270)
            left_right_flip_image_Crop_270, left_right_flip_gt_Crop_270 = randomCrop(left_right_flip_image_Rotation_270, left_right_flip_gt_Rotation_270)

            image.save(save_aug_images_path + dataset + '_' + str(i) + '.jpg')
            image_Enhance.save(save_aug_images_path + dataset + '_' + str(i) + '_e.jpg')
            image_Gaussian.save(save_aug_images_path + dataset + '_' + str(i) + '_g.jpg')
            image_Crop.save(save_aug_images_path + dataset + '_' + str(i) + '_c.jpg')
            image_Crop_90.save(save_aug_images_path + dataset + '_' + str(i) + '_c_90.jpg')
            image_Crop_180.save(save_aug_images_path + dataset + '_' + str(i) + '_c_180.jpg')
            image_Crop_270.save(save_aug_images_path + dataset + '_' + str(i) + '_c_270.jpg')
            image_Rotation_90.save(save_aug_images_path + dataset + '_' + str(i) + '_r_90.jpg')
            image_Rotation_180.save(save_aug_images_path + dataset + '_' + str(i) + '_r_180.jpg')
            image_Rotation_270.save(save_aug_images_path + dataset + '_' + str(i) + '_r_270.jpg')
            
            left_right_flip_image.save(save_aug_images_path + dataset + '_' + str(i) + '_lf.jpg')
            left_right_flip_image_Enhance.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_e.jpg')
            left_right_flip_image_Gaussian.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_g.jpg')
            left_right_flip_image_Crop.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_c.jpg')
            left_right_flip_image_Crop_90.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_c_90.jpg')
            left_right_flip_image_Crop_180.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_c_180.jpg')
            left_right_flip_image_Crop_270.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_c_270.jpg')
            left_right_flip_image_Rotation_90.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_r_90.jpg')
            left_right_flip_image_Rotation_180.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_r_180.jpg')
            left_right_flip_image_Rotation_270.save(save_aug_images_path + dataset + '_' + str(i) + '_lf_r_270.jpg')

            gt.save(save_aug_gts_path + dataset + '_' + str(i) + '.png')
            gt_Enhance.save(save_aug_gts_path + dataset + '_' + str(i) + '_e.png')
            gt_Gaussian.save(save_aug_gts_path + dataset + '_' + str(i) + '_g.png')
            gt_Crop.save(save_aug_gts_path + dataset + '_' + str(i) + '_c.png')
            gt_Crop_90.save(save_aug_gts_path + dataset + '_' + str(i) + '_c_90.png')
            gt_Crop_180.save(save_aug_gts_path + dataset + '_' + str(i) + '_c_180.png')
            gt_Crop_270.save(save_aug_gts_path + dataset + '_' + str(i) + '_c_270.png')
            gt_Rotation_90.save(save_aug_gts_path + dataset + '_' + str(i) + '_r_90.png')
            gt_Rotation_180.save(save_aug_gts_path + dataset + '_' + str(i) + '_r_180.png')
            gt_Rotation_270.save(save_aug_gts_path + dataset + '_' + str(i) + '_r_270.png')

            left_right_flip_gt.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf.png')
            left_right_flip_gt_Enhance.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_e.png')
            left_right_flip_gt_Gaussian.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_g.png')
            left_right_flip_gt_Crop.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_c.png')
            left_right_flip_gt_Crop_90.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_c_90.png')
            left_right_flip_gt_Crop_180.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_c_180.png')
            left_right_flip_gt_Crop_270.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_c_270.png')
            left_right_flip_gt_Rotation_90.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_r_90.png')
            left_right_flip_gt_Rotation_180.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_r_180.png')
            left_right_flip_gt_Rotation_270.save(save_aug_gts_path + dataset + '_' + str(i) + '_lf_r_270.png')

if __name__=='__main__':
    image_path = './dataset/'
    # datasets = ['EORSSD', 'ORSSD']   
    datasets = ['EORSSD'] 
    save_path = 'data_aug'
    data_aug(image_path, datasets, save_path)