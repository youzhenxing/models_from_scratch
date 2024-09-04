import matplotlib.pyplot as plt

def plot_img_and_mask(img,mask):
    classes = mask.max() + 1
    fig,ax = plt.subplots(1, classes + 1)
    ax[0].set_title('input images')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_axis_off()
        ax[i+1].imshow(mask==i)
    plt.xticks([]), plt.yticks([])
    plt.show()