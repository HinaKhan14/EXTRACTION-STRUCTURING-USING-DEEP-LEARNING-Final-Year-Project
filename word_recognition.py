import cv2
import pytesseract
from pytesseract import Output
import prediction

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sort(cont):
    A = []
    for t in range(len(cont)):
        A.append(cont[t][0][0][0])
    print("unsorting cont: ", A)
    ###################################
    # for i in range(len(A)):
    #     # Find the minimum element in remaining
    #     # unsorted array
    #     min_idx = i
    #     for j in range(i + 1, len(A)):
    #         if A[min_idx] >= A[j]:
    #             min_idx = j
    #
    #     # Swap the found minimum element with
    #     # the first element
    #     A[i], A[min_idx] = A[min_idx], A[i]
    # new_sort_list = []
    # m = 0
    # cont_length = len(cont)
    # print("length of cont", len(cont))
    # i = 0;
    # while (i <= len(A) and m < len(cont)):
    #     if (cont[m][0][0][0] == A[i]):
    #         new_sort_list.append(cont[m])
    #         print("new sorted list m ", i, " ", new_sort_list[i])
    #         i += 1
    #         m = 0
    #     m += 1
    # print("sorted A", A)
    # print("new sorted list", new_sort_list)

    ################################
    # Traverse through all array elements
    for i in range(len(cont)):
        # Find the minimum element in remaining
        # unsorted array
        min_idx = i
        for j in range(i + 1, len(cont)):
            if cont[min_idx][0][0][0] > cont[j][0][0][0]:
                min_idx = j

        # Swap the found minimum element with
        # the first element
        cont[i], cont[min_idx] = cont[min_idx], cont[i]

        # Driver code to test above
    # print("Sorted array")
    # for i in range(len(A)):
    print("sorted cont: ", cont)
    return cont


def digit_detection(img_path,img_num):
    img = img_path

    # --------------PRINT IMAGE WE GET FROM CROPPED IMAGE ------#
    # plotting = plt.imshow(img, cmap='gray')
    # plt.show()
    # ----------------------------------------------------------#

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------SHOW GRAY IMAGE ----------------------#
    # plotting = plt.imshow(gray, cmap='gray')
    # plt.show()
    # -------------------------------------------------------#

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # -----------------SHOW BINARY INVERT IMAGE --------------#
    # plotting = plt.imshow(thresh1, cmap='gray')
    # plt.show()
    # --------------------------------------------------------#

    # --- choosing the right kernel
    # --- kernel size of 1 rows (to join dots above letters 'i' and 'j')
    # --- and 3 columns to join neighboring letters in words and neighboring words
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # ---------------SHOW DILATED IMAGE ---------------------#
    # plotting = plt.imshow(dilation, cmap='gray')
    # plt.show()
    # ------------------------------------------------------#

    # ---Finding contours ---
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # ---Sort contours ---
    contours = sort(contours)

    # Create list box to store all boxes in
    box = []
    out = []
    output_of_CNN = []
    i = 0
    count = 0


    for c in contours:
        i = i + 1
        x, y, w, h = cv2.boundingRect(c)
        # and w > 3 * h
        if w > 1 and h > 7:
            # if (w<1000 and h<500) and (w > 20 and h > 20):
            count += 1
            cropped = img[y:y + h, x:x + w]
            box.append([x, y, w, h])

            # --------PRINT CROPPED IMAGE ------------#
            # plotting = plt.imshow(cropped, cmap='gray')
            # plt.show()
            # ---------------------------------------#

            gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # ---------BINARY IMAGE OF CROPPED IMAGE -------#
            # plotting = plt.imshow(threshold_img, cmap='gray')
            # plt.show()
            # -----------------------------------------------#

            #################################
            ####           ############             ############

            output_of_CNN.append(prediction.digit_or_alphabet_predictor(threshold_img, img_num))

            ####            #########               ############
            img_num += 1

            # cv2.imshow('threshold_image ', threshold_img)
            # cv2.waitKey(0)
            custom_config = r'--oem 3 --psm 6'
            out.append(pytesseract.image_to_string(threshold_img, output_type=Output.STRING, config=custom_config,
                                                   lang='eng'))

            # print("pytesseract output", out)
            # print("CNN output ", output_of_CNN)
            ###############################################

        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # plotting = plt.imshow(rect, cmap='gray')
        # plt.show()
    listToStr = ''.join([str(elem) for elem in output_of_CNN])
    print("CNN string", listToStr)
    StrToInt = int(listToStr)
    return StrToInt
    # return listToStr
#
