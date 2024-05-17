import cv2
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Shivashanmuga\Downloads\Tesseract-OCR 1\Tesseract-OCR\tesseract.exe'

# Load the image from the given path
image_path = cv2.imread(r'C:\Users\Shivashanmuga\Downloads\11zon_PDF-to-JPG (1)\11zon_PDF-to-JPG\invoice_2001665\invoice_2001665_1.jpg')

# Convert the image colors from BGR to RGB as matplotlib uses RGB format
image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

# Perform OCR with layout analysis using the 'hOCR' output format
image_hocr = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT,
                                       config='--psm 3 -c tessedit_create_hocr=2')

# Convert the image_hocr dictionary to a list of dictionaries, where each
# dictionary contains the information for a single word
words = []
for i, word_info in enumerate(image_hocr['text']):
    if word_info:
        left = image_hocr['left'][i]
        top = image_hocr['top'][i]
        width = image_hocr['width'][i]
        height = image_hocr['height'][i]
        words.append({
            'text': word_info,
            'left': left,
            'top': top,
            'width': width,
            'height': height,
        })

# Extract the text and coordinate information from the list of dictionaries
text = []
coords = []
for word in words:
    text.append(word['text'])
    coords.append((word['left'], word['top']))
    # print(coords)

# Let the user select the ROIs for the crop operation
print("Select the ROIs (type 'q' to quit)")
cropped_images = []
cropped_text_lists = []
i = 0
while True:
    r = cv2.selectROI(image_rgb, False)
    if r[0] == 0 and r[1] == 0 and r[2] == 0 and r[3] == 0:
        break
    else:
        x1, y1, w, h = r
        crop = image_path[y1:y1 + h, x1:x1 + w]

        # Display the cropped image
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')
        plt.show()

        # Perform OCR on the cropped image
        cropped_text = pytesseract.image_to_string(crop, config = '--psm 3 -c tessedit_create_hocr=2')

        # Store the detected text in the current list
        cropped_text_lists.append([])
        for j, line in enumerate(cropped_text.split('\n')):
            if line:
                cropped_text_lists[-1].append(line)

        combine_list = []
        # Append each inner list to the combine_list
        for inner_list in cropped_text_lists:
            combine_list.append(inner_list)
        # print("combine lists:\n", combine_list)

        # Define the number of columns
        n = 4

        # # Create a list of column names
        # column_names = ['col_{}'.format(i) for i in range(1, n + 1)]
        # columns = column_names

        # Create a DataFrame with 'n' columns
        # df = pd.DataFrame()

        df = pd.DataFrame(combine_list)
        df_transposed = df.transpose()
        # print(df)
        print(df_transposed)
        df_transposed.to_csv('output.csv')

        # Store the cropped image for display later
        cropped_images.append(crop)
        # print(f'Detected Text: {cropped_text_lists[-1]}\n')

print("Selected ROIs and their corresponding text:")
for i, (image, text_list) in enumerate(zip(cropped_images, cropped_text_lists)):
    print(f"\nROI {i + 1}:")
    print("\nImage:")
    print(cropped_text_lists)
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.show()

# import cv2
# import pytesseract
# import matplotlib.pyplot as plt
#
# # Load the image from the given path
# image = cv2.imread(r'C:\Users\Shivashanmuga\Downloads\invoice_2001665 1_page-0001.jpg')
#
# # Convert the image colors from BGR to RGB as matplotlib uses RGB format
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Display the original image
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# plt.title('Original Image')
# plt.show()
#
# # Perform OCR on the image using Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Shivashanmuga\Downloads\Tesseract-OCR 1\Tesseract-OCR\tesseract.exe'
#
# # Perform OCR with layout analysis using the 'Text' output format
# image_string = pytesseract.image_to_string(image, config='--psm 3 -c tessedit_create_hocr=2')
#
# # Print the extracted text
# print(image_string)
#
# # Display the detected text and their coordinates in another image
# image_detected_text = image.copy()
# x, y = 10, 10
# for word in image_string.split('\n'):
#     if word:
#         cv2.putText(image_detected_text, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#         y += 15
#
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(image_detected_text, cv2.COLOR_BGR2RGB))
# plt.title('Detected Text')
# plt.show()