import cv2
import matplotlib.pyplot as plt
import numpy as np

path1 = "Image Preprocessing Test/hand1_h_dif_seg_4_cropped.jpeg"
path2 = "Image Preprocessing Test/WIN_20220201_09_16_13_Pro.jpg"

#reading image in gray scale.
test_img1 = cv2.imread(path1,0 )
test_img2 = cv2.imread(path2,0)

test_img1 = cv2.resize(test_img1, dsize=(60, 60))
test_img2 = cv2.resize(test_img2, dsize=(60, 60))


def normalimages():
    normalimages1 = cv2.imread(path1)
    normalimages2 = cv2.imread(path2)

    normalimages1 = cv2.resize(normalimages1, dsize=(60, 60))
    normalimages2 = cv2.resize(normalimages2, dsize=(60, 60))

    normalimage1 = cv2.cvtColor(normalimages1, cv2.COLOR_BGR2RGB)
    normalimage2 = cv2.cvtColor(normalimages2, cv2.COLOR_BGR2RGB)

    plt.imshow(normalimage1)
    plt.title("No Preprocessing Dataset Image")
    plt.show()
    plt.imshow(normalimage2)
    plt.title("No Preprocessing Non Dataset Image")
    plt.show()


def agtmorphology():
    #Adaptive gaussian thresholding.
    #Notes - good for images with diffrent light levels.
    agt_image1 = cv2.adaptiveThreshold(test_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)

    agt_image2 = cv2.adaptiveThreshold(test_img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)

    #Applying morpholohy- results: thickens the noise and does not remove it.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    morphology_img1 = cv2.morphologyEx(agt_image1, cv2.MORPH_OPEN, kernel,iterations=1)
    morphology_img2 = cv2.morphologyEx(agt_image2, cv2.MORPH_OPEN, kernel,iterations=1)

    plt.imshow(agt_image1, cmap="gray")
    plt.title("Adaptive Gaussian Thresholding")
    plt.xlabel("Dataset Image")
    plt.show()
    plt.imshow(agt_image2, cmap="gray")
    plt.title("Adaptive Gaussian Thresholding")
    plt.xlabel("Not Dataset Image")
    plt.show()

    plt.imshow(morphology_img1, cmap="gray")
    plt.title("Adaptive Gaussian Thresholding With Morphology")
    plt.xlabel("Dataset Image")
    plt.show()
    plt.imshow(morphology_img2, cmap="gray")
    plt.title("Adaptive Gaussian Thresholding With Morphology")
    plt.xlabel("Not Dataset Image")
    plt.show()


def divideotsu():
    #Grayscale -> Blur -> Divide -> Otsu Thresholding

    #Results:
    blur1 = cv2.GaussianBlur(test_img1, (0, 0), sigmaX=33, sigmaY=33)
    divide1 = cv2.divide(test_img1, blur1, scale=255)
    thresh1 = cv2.threshold(divide1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    blur2 = cv2.GaussianBlur(test_img2, (0,0), sigmaX=33, sigmaY=33)
    divide2 = cv2.divide(test_img2, blur2, scale=255)
    thresh2 = cv2.threshold(divide2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


    plt.imshow(thresh1,cmap="gray")
    plt.title("Blur/Divide/OtsuThresholding")
    plt.xlabel("Dataset Image")
    plt.show()

    plt.imshow(thresh2,cmap="gray")
    plt.title("Blur/Divide/OtsuThresholding")
    plt.xlabel("Not Dataset Image")
    plt.show()

def comparison():
    Test1=cv2.adaptiveThreshold(test_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                             cv2.THRESH_BINARY, 11, 2)

    blur2 = cv2.GaussianBlur(test_img2, (0,0), sigmaX=33, sigmaY=33)
    divide2 = cv2.divide(test_img2, blur2, scale=255)
    thresh2 = cv2.threshold(divide2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    plt.imshow(Test1)
    plt.show()

    plt.imshow(thresh2)
    plt.show()

# for i in range(1):
# normalimages()
# agtmorphology()
# divideotsu()
# comparison()





