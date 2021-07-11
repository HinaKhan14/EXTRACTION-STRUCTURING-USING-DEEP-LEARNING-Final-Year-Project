import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
i =0
predicted_word = []
# model = load_model('mnist_for_digits.h5')
model = load_model('mnist_for_digits.h5')
labels_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'A', 11: 'B', 12: 'C', 13: 'D',
               14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
               25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
               36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
               47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
               58: 'w', 59: 'x', 60: 'y', 61: 'z'}


def digit_or_alphabet_predictor(img, d):
    # plotting = plt.imshow(img, cmap='gray')
    # plt.show()
    img = cv2.resize(img, (32, 32))
    threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # plotting = plt.imshow(threshold_img, cmap='gray')
    # plt.show()
    filename = "images/file_%d.jpg" %d
    cv2.imwrite(filename, threshold_img)
    # print(d)
    # cv2.imwrite("template{0}.jpg".format(img_num), threshold_img)
    im_final = threshold_img.reshape(1, 32, 32, 1)

    # filename = 'savedImage.jpg'
    # cv2.imwrite(filename, im_final)
    ans = model.predict(im_final)
    print("model prediction: ", ans)
    ans = np.argmax(ans, axis=1)[0]

    # key = labels_dict[ans]
    # predicted_word.append(key)
    # print(key)
    print(ans)
    # print(predicted_word)
    # return key
    return ans