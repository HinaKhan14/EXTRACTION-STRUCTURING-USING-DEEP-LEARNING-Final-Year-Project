import cv2
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import word_recognition

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img_num = 0

def sort(cont):
    A = []
    for t in range(len(cont)):
        A.append(cont[t][0][0][0])
    print("unsorting cont: ", A)
    # Traverse through all array elements
    #################################
    for i in range(len(cont)):
        # Find the minimum element in remaining
        # unsorted array
        min_idx = i
        for j in range(i + 1, len(cont)):
            if cont[min_idx][0][0][0] >= cont[j][0][0][0]:
                min_idx = j

        # Swap the found minimum element with
        # the first element
    cont[i], cont[min_idx] = cont[min_idx], cont[i]
    #########################
    # swapped = True
    # while swapped:
    #     swapped = False
    #     for i in range(len(cont)-1):
    #         if cont[i][0][0][0] > cont[i+1][0][0][0]:
    #             # Swap the elements
    #             cont[i], cont[i+1] = cont[i+1], cont[i]
    #             # Set the flag to True so we'll loop again
    #             swapped = True
###########################
    # Driver code to test above
    # print("Sorted array")
    # for i in range(len(A)):
    #     print("sorted cont: ", cont)
    #

    return cont


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

        # print(boundingBoxes)
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)


def table_detection(img_path):
    img = cv2.imread(img_path)
    # ----- PRINT ORIGINAL IMAGE ------- #
    # plotting = plt.imshow(img, cmap='gray')
    # plt.show()
    # ---------------------------------- #
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------ PRINT GRAY IMAGE -----------#
    # plotting = plt.imshow(img_gray, cmap='gray')
    # plt.show()
    # ------------------------------------#

    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plotting = plt.imshow(img_bin, cmap='gray')
    plt.show()
    img_bin = cv2.bitwise_not(img_bin)

    # ---- PRINT BINARY IMAGE -----------#
    # plotting = plt.imshow(img_bin, cmap='gray')
    # plt.show()
    # ------------------------------------#

    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)

    # -------PRINT VERTICAL LINES --------#
    # plotting = plt.imshow(im_temp1, cmap='gray')
    # plt.show()
    # ------------------------------------#

    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    kernel_length_h = (np.array(img_gray).shape[1]) // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)

    # ------PRINT HORIZONTAL LINES --------#
    # plotting = plt.imshow(im_temp2, cmap='gray')
    # plt.show()
    # ------------------------------------#

    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)

    # -----PRINT JOINING OF HORIZONTAL AND VERTICAL LINES---- #
    # plotting = plt.imshow(table_segment, cmap='gray')
    # plt.show()
    # ---------------------------------------------------------#

    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)

    # -------------PRINT BITWISE NOT(INVERT) IMAGE----------#
    # plotting = plt.imshow(table_segment, cmap='gray')
    # plt.show()
    # ------------------------------------------------------#

    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    # ------------THRESHOLD BITWISE IMAGE ----------------#
    # plotting = plt.imshow(table_segment, cmap='gray')
    # plt.show()
    # ----------------------------------------------------#

    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("image contours", contours)
    contour = sort(contours)

    # Sort all the contours by top to bottom.
    # contour, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Creating a list of heights for all detected boxes
    # heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    # Get mean of heights
    # mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    out = []
    list_of_CNN_words = []
    i = 0
    count = 0
    n = 0

    for c in contour:

        i = i + 1
        x, y, w, h = cv2.boundingRect(c)
        # x, y, w, h = contours[c]
        # and w > 3 * h
        # if ((w > 50 and h > 20) and w > 3 * h):
        if (w < img_width and h < img_height) and (w > 20 and h > 20):
            count += 1
            cropped = img[y:y + h, x:x + w]
            box.append([x, y, w, h])

            # ---------  PRINT CROPPED BOX IMAGE-------#
            # plotting = plt.imshow(cropped, cmap='gray')
            # plt.show()
            # -----------------------------------------#

            if count > 1:  # THIS IS BECAUSE 1ST COUNT IS FOR THE WHOLE TABLE
                gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # --------------CROPPED IMAGE THRESHOLD TO BINARY IMAGE -------#
                # plotting = plt.imshow(threshold_img, cmap='gray')
                # plt.show()
                # -------------------------------------------------------------#

                # ------ PASS THE CROPPED IMAGE TO A CLASS WHICH SPLIT EACH WORD ----#
                list_of_CNN_words.append(word_recognition.digit_detection(cropped,img_num))
                # ---------------------------------------------------------------------#

                # --------------USE PYTESSERACT TO CONVERT INTO IMAGE TO STRING -------#
                custom_config = r'--oem 3 --psm 6'
                out.append(pytesseract.image_to_string(threshold_img, output_type=Output.STRING, config=custom_config,
                                                       lang='eng'))
                # print("first alpha",out[n][0])
                #
                # if out[n][0].isdigit():
                #     # ------ PASS THE CROPPED IMAGE TO A CLASS WHICH SPLIT EACH WORD ----#
                #     list_of_CNN_words.append(word_recognition.digit_detection(cropped, img_num))
                #     # ---------------------------------------------------------------------#
                # else:
                #     list_of_CNN_words.append(out[n])
                #     print(list_of_CNN_words)
                # n += 1
                # print("out", out)
                # print(list_of_CNN_words)
                # ----------------------------------------------------------------------#
            ###############################################
            if True:
                cv2.imwrite("./results/cropped/crop_" + str(count) + "__" + img_path.split('/')[-1], cropped)
        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # plotting = plt.imshow(rect, cmap='gray')
        # plt.show()
    if True:
        cv2.imwrite("./results/table_detect/table_detect__" + img_path.split('/')[-1], table_segment)
        cv2.imwrite("./results/bb/bb__" + img_path.split('/')[-1], img)
    print("list of CNN words", list_of_CNN_words)
    # rows and cols
    row = []
    col = []
    y = 0
    x = 0
    x_count = 0
    y_count = 0
    row_greater_num = []
    # for i in range(len(box)):
    #     if box[x][y] > box[x + 1][y] and i!=0:
    #         x_count += 1
    #         print(x_count)
    #     elif i!=0:
    #         x_count = 0
    #         y_count += 1
    #         print(y_count)
    c = 0
    print("box", box)
    # to find number of rows and cols for excel sheet
    for x in range(len(box) - 1):
        if box[x][y] > box[x + 1][y] and (i > 1 or (x == 0 and y_count > 0)):
            x_count += 1
            row.append(x_count)
        elif x > 1:
            row.append(x_count + 1)
            row_greater_num.append(x_count + 1)
            x_count = 0
            y_count += 1
            col.append(y_count)
    row.append(x_count + 1)
    row_greater_num.append(x_count + 1)
    col.append(y_count + 1)

    print("row", len(row))
    print("col", col)
    print(row_greater_num)
    start = 0
    end = 0
    rows = []
    image_to_string_list = []
    for i in range(len(row)-1):
        if row[i] > row[i + 1]:
            end = i + 1
            image_to_string_list.append(list_of_CNN_words[start:end])
            rows.append(row[start:end])
            # print("row start end ",row[start:end], i)
            start = end
            i = 0
    image_to_string_list.append(list_of_CNN_words[start:len(row)])

    sorted_list = []
    print("image to text ", image_to_string_list[len(image_to_string_list) - 1])
    print(image_to_string_list)


    sorted_list.append(image_to_string_list[-1])
    for i in range(0, len(image_to_string_list)):
        for j in reversed(range(0, len(image_to_string_list[i]))):
            sorted_list.append(image_to_string_list[j])

    print("sorted list", sorted_list)

    print("list in rows ", rows)
    print(len(rows))
    print("length of image to string: ", len(image_to_string_list))

    print("img 0", image_to_string_list)
    print(len(image_to_string_list))
    new_image_to_string = []
    for i in range(0, len(image_to_string_list)):
        for j in reversed(range(len(image_to_string_list[i]))):
            new_image_to_string.append(sorted_list[i][j])

    print("final to save",new_image_to_string)
    # for i in range(len(new_image_to_string)):
    #     new_image_to_string[i] = new_image_to_string[i].replace('\x0c', '')
    # print(new_image_to_string)
    workbook = xlsxwriter.Workbook('write_data.xlsx')
    worksheet = workbook.add_worksheet('Sheet1')
    n = 0
    print(new_image_to_string[-1])
    text_num = 0
    print("len of col", len(col))
    print("len of rows", len(rows))
    # print("len of row", len(row))
    for i in range(len(col)):
        if n < len(rows):
            for j in range(len(rows[n])):
                worksheet.write(i, j, new_image_to_string[text_num])  # Writes
                text_num += 1
    n += 1
    workbook.close()

#
# table_detection(img_path='mytable.jpg')
