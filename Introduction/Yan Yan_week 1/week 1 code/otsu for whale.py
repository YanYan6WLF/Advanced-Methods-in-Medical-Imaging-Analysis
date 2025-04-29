import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def otsu(img: np.ndarray) -> float:
    hist=np.bincount(img.flatten(), minlength=256)
    total_pixel_num=img.size
    num_background=0
    num_foreground=0
    current_max_inter_variance=0
    threshold=0
    sum_pixel_value = np.sum(np.arange(256) * hist)
    inter_variance_list=[]
    intra_variance_list = []
    sum_pixel_back=0

    for i in range(256):
        num_background=num_background+hist[i]
        if num_background==0:
            continue
        num_foreground=total_pixel_num-num_background
        if num_foreground==0:
            break
        sum_pixel_back =sum_pixel_back+ i*hist[i]
        sum_pixel_fore=sum_pixel_value-sum_pixel_back
        mean_pixel_back=sum_pixel_back/num_background
        mean_pixel_fore=sum_pixel_fore/num_foreground


        intra_variance = num_background * (mean_pixel_back - mean_pixel_back ** 2) + num_foreground * (mean_pixel_fore - mean_pixel_fore ** 2)
        intra_variance_list.append(intra_variance)
        
        inter_variance = num_background * num_foreground * (mean_pixel_back - mean_pixel_fore) ** 2
        inter_variance_list.append(inter_variance)
        

        if inter_variance > current_max_inter_variance:
            current_max_inter_variance=inter_variance
            threshold=i 

   

    return threshold, intra_variance_list, inter_variance_list



    

if __name__ == "__main__":
    I = Image.open('whale.png')
    grey_img=I.convert('L')
    img = np.array(grey_img)
    print('Image resoluttion', img.shape)
    threshold, intra_variance_list, inter_variance_list = otsu(img)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Histogram of Image')
    plt.hist(img.flatten(), bins=256)
    plt.axvline(x=threshold, color='k', linestyle='--')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.title('Intra-class Variance Curve')
    plt.plot(range(len(intra_variance_list)), intra_variance_list)
    plt.xlabel('Threshold')
    plt.ylabel('Intra-class Variance')

    plt.subplot(2, 2, 3)
    plt.title('Inter-class Variance Curve')
    plt.plot(range(len(inter_variance_list)), inter_variance_list)
    plt.xlabel('Threshold')
    plt.ylabel('Inter-class Variance')

    otsu_img = (img > threshold) * 255
    plt.subplot(2, 2, 4)
    plt.title(f'Thresholded Image with Threshold {threshold}')
    plt.imshow(otsu_img, cmap='gray')

    plt.tight_layout()
    plt.show()

    print(f"The calculated threshold is: {threshold}")







 






 