import numpy as np
import cv2
from keras.datasets import mnist
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
def digit_recog():
    model = keras.models.load_model(r'C:\Users\Khan\PycharmProjects\part 3\my_model')
    # model = tf.keras.models.load_model('/tmp/model')
    # model = load_model('trained_model.h5')
    img = cv2.imread('8.jpg')
    plotting = plt.imshow(img, cmap='gray')
    plt.show()
    img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plotting = plt.imshow(img_gray, cmap='gray')
    plt.show()
    img_gray_u8 = img_as_ubyte(img_gray)
    plotting = plt.imshow(img_gray_u8, cmap='gray')
    plt.show()
    (thresh, im_binary) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_resized = cv2.resize(im_binary, (28, 28))
    im_gray_invert = 255 - img_resized
    plotting = plt.imshow(im_gray_invert, cmap='gray')
    plt.show()
    im_final = im_gray_invert.reshape(1, 28, 28, 1)
    ans = model.predict(im_final)
    ans = np.argmax(ans, axis=1)[0]
    print(ans)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(x_train[i], cmap='binary', interpolation='nearest')
#     ax.text(0.05, 0.05, str(y_train[i]), transform=ax.transAxes, color='green')
# plt.show()
#


# kernel_length_v = (np.array(img_gray).shape[1]) // 120
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
#
# im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
# plotting = plt.imshow(im_temp1, cmap='gray')
# plt.show()


# width = 640
# height = 480
# cameraNo = 0
#
# cap = cv2.VideoCapture(cameraNo)
# cap.set(3, width)
# cap.set(4, height)
#
# model = load_model('trained_model.h5')
# while True:
#     success, im_orig = cap.read()
#
#     img_gray = rgb2gray(im_orig)
#     img_gray_u8 = img_as_ubyte(img_gray)
#     cv2.imshow("Window", img_gray_u8)
#     (thresh, im_binary) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     img_resized = cv2.resize(im_binary, (28, 28))
#
#     im_gray_invert = 255 - img_resized
#     cv2.imshow("invert image", im_gray_invert)
#     im_final = im_gray_invert.reshape(1, 28, 28, 1)
#     ans = model.predict(im_final)
#     ans = np.argmax(ans, axis=1)[0]
#     print(ans)


#     cv2.putText(im_orig, 'Predicted Digit : ' + str(ans), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
#     cv2.imshow("Original Image", im_orig)
#     if cv2.waitKey(1) and 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
#                          gri  dspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(x_train[i], cmap='binary', interpolation='nearest')
#     ax.text(0.05, 0.05, str(y_train[i]), transform=ax.transAxes, color='green')
# plt.show()
