import cv2
import numpy as np
import imutils
import pytesseract
from datetime import datetime
import openpyxl
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


excel_file_path = r"C:\Users\Deepanshu Singhal\PycharmProjects\Python OCR"

def load_images_and_labels(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_path = os.path.join(directory, filename)
            label = parse_xml(xml_path)
            if label is not None:
                img_filename = filename.replace(".xml", ".jpg")
                img_path = os.path.join(directory, img_filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (20, 20))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)


def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()


        object_element = root.find('object')

        if object_element is not None:

            name_element = object_element.find('name')


            if name_element is not None:

                label = name_element.text.strip()
                return label
            else:
                print(f"No <name> element found in {xml_path}")
                return None
        else:
            print(f"No <object> element found in {xml_path}")
            return None

    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None


dataset_folder = r'C:\Users\Deepanshu Singhal\PycharmProjects\Python OCR\voc_plate_dataset\voc_plate_dataset\Annotations'
images, labels = load_images_and_labels(dataset_folder)


unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Unique Labels:", unique_labels)
print("Label Counts:", label_counts)


if len(unique_labels) < 2:
    print("Not enough unique labels to perform train-test split.")
else:

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, train_size=0.8, random_state=42)


    knn_classifier = KNeighborsClassifier(n_neighbors=3)


    knn_classifier.fit(X_train, y_train)


    image = cv2.imread('car.jpeg')
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)


    image = imutils.resize(image, width=500)

    #
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Scaled Image", gray)
    cv2.waitKey(0)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Smoothed Image", gray)
    cv2.waitKey(0)


    edged = cv2.Canny(gray, 170, 200)
    cv2.imshow("Canny Edge", edged)
    cv2.waitKey(0)


    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Canny After Contouring", image1)
    cv2.waitKey(0)


    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]


    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Top 30 Contours", image2)
    cv2.waitKey(0)


    NUMBERPLATECOUNT = None
    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)

        if len(approx) == 4:
            NUMBERPLATECOUNT = approx
            x, y, w, h = cv2.boundingRect(i)
            crp_img = image[y:y + h, x:x + w]
            cv2.imwrite('output.jpg', crp_img)
            break


    cv2.drawContours(image, [NUMBERPLATECOUNT], -1, (0, 255, 0), 3)
    cv2.imshow("Final Image", image)
    cv2.waitKey(0)


    cv2.imshow("License Plate Region", crp_img)
    cv2.waitKey(0)


    text = pytesseract.image_to_string(crp_img, lang='eng')
    print("THE LICENSE PLATE NUMBER IS:", text)


    crp_img_resized = cv2.resize(crp_img, (20, 20))
    predicted_label = knn_classifier.predict(crp_img_resized.flatten().reshape(1, -1))[0]

    print("Predicted License Plate Number:", predicted_label)


    wb = openpyxl.load_workbook(excel_file_path)


    sheet = wb.active


    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    sheet.append([current_datetime,  predicted_label])


    wb.save(excel_file_path)


    cv2.destroyAllWindows()


def load_images_and_labels(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_path = os.path.join(directory, filename)

            label = parse_xml(xml_path)
            if label is not None:
                img_filename = filename.replace(".xml", ".jpg")
                img_path = os.path.join(directory, img_filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (20, 20))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)

