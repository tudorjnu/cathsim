import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
images_path = os.path.join(cwd, "data", "images")

for filename in os.listdir(images_path)[::-1]:
    image_pair = np.load(os.path.join(images_path, filename))
    image_A = image_pair[:, :, 0]
    image_B = image_pair[:, :, 1]
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_A)  # , cmap="gray")
    axarr[1].imshow(image_B)  # , cmap="gray")
    print(np.true_divide(image_B.sum(), (image_B != 0).sum()))
    # plt.show()
