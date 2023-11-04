import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

# Calculates the energy matrix of the input image using gradients in x and y directions
# outputs the resulting matrix
def compute_energy_matrix(img):
    img_8bit = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

# goal is to get the lowest costing seam using dynammic programming
def find_lowest_cost_seam(energy_matrix):
    M = energy_matrix.copy()
    rows, cols = M.shape

    # geos through the matrix given in the parameter to get the cost
    for row in range(1, rows):
        for col in range(cols):
            if col == 0:
                M[row, col] += min(M[row - 1, col], M[row - 1, col + 1])
            elif col == cols - 1:
                M[row, col] += min(M[row - 1, col], M[row - 1, col - 1])
            else:
                M[row, col] += min(M[row - 1, col - 1], M[row - 1, col], M[row - 1, col + 1])

    return M

# takes a matrix m, and runs through via backtracking to get the optimal seam from 
# the cumualtive cost of the matrix below
def get_optimal_seam(M):
    seam = []
    rows, cols = M.shape
    idx = np.argmin(M[-1])
    seam.append(idx)

    for row in range(rows-2, -1, -1):
        prev_idx = seam[-1]
        if prev_idx == 0:
            idx_offset = np.argmin(M[row, :2])
        elif prev_idx == cols - 1:
            idx_offset = np.argmin(M[row, -2:]) - 1
        else:
            idx_offset = np.argmin(M[row, prev_idx-1:prev_idx+2]) - 1
        seam.append(prev_idx + idx_offset)
    seam.reverse()
    return seam

# runs through the image, and using a given seam, removes it from the image via a loop
# will output the new image with the seam removed.
def remove_seam(img, seam):
    output = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]))
    for row in range(img.shape[0]):
        output[row, :seam[row]] = img[row, :seam[row]]
        output[row, seam[row]:] = img[row, seam[row] + 1:]
    return output

# seamCarve uses above functions to change the image to desired demensions 
def SeamCarve(input, widthFac, heightFac, mask):
    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    inSize = input.shape
    size   = (int(widthFac*inSize[1]), int(heightFac*inSize[0]))
    num_seams_to_remove_width = inSize[1] - size[0]
    num_seams_to_remove_height = inSize[0] - size[1]

    if widthFac < 1:
        for _ in range(num_seams_to_remove_width):
            energy_matrix = compute_energy_matrix(input)
            M = find_lowest_cost_seam(energy_matrix)
            seam = get_optimal_seam(M)
            input = remove_seam(input, seam)

    if heightFac < 1:
        input = np.rot90(input)
        for _ in range(num_seams_to_remove_height):
            energy_matrix = compute_energy_matrix(input)
            M = find_lowest_cost_seam(energy_matrix)
            seam = get_optimal_seam(M)
            input = remove_seam(input, seam)
        input = np.rot90(input, k=-1)

    return input, size

inputDir = '../img/'
outputDir = '../Results/'

# variables used to change the dimensions of the image
widthFac = .5
heightFac = 1
N = 4

for index in range(1, N + 1):
    input, mask = Read(str(index).zfill(2), inputDir)
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    plt.imsave("{}/resultHW_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)

widthFac = 1
heightFac = .5
N = 4
for index in range(1, N + 1):
    input, mask = Read(str(index).zfill(2), inputDir)
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    plt.imsave("{}/resultHH_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)
