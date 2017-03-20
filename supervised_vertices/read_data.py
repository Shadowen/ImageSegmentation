import json
import numpy as np
import glob
import scipy.misc
import matplotlib.pyplot as plt


def read(directory):
    filepaths = glob.glob(directory + '/info/*.info')
    images = np.array([len(filepaths), 224, 224, 3])
    segmentations = [None for _ in range(len(filepaths))]

    for i, filepath in enumerate(filepaths):
        with open(filepath) as info_file:
            contents = json.load(info_file)
            # {boundaries, patch_id, mult_poly, grid, bbox, category_id, patch_path, segmentation, image_id}
            segmentations[i] = contents['segmentation']
            assert (len(contents['patch_path']) == 1)  # idk why there's an extra array nesting here...
            patch_path = contents['patch_path'][0]
        with open(patch_path, 'rb') as image_file:
            images[i] = scipy.misc.imread(image_file)

    return images, segmentations


if __name__ == '__main__':
    images, segmentations = read('/home/wesley/data/train/')
    # images, segmentations = read('/ais/gobi4/wiki/polyrnn/data/shapes_texture/train/')

    plt.imshow(images[0])
