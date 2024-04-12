# Testing inference images
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
page_path = r"D:\iam\raw_dataset\page1"
# Source path, the images and labels are located (For Training)
source_path = r"C:\Datasets\iam_words\words"
# Initialize a label path and an empty label list to store valid transcriptions (For Training)
base_path = r"C:\Datasets\iam_words\words.txt"
# Load_path of Scanned Image (Test Data Input Pipeline) (for Page Prediction)
extracted_image_path = r"D:\iam\extracted_word_imgs_1"
# Number of epochs
epochs = 5
# Define the filepath to save the model
filepath = "D:/iam/epochs/model_epoch_{epoch:02d}.h5"
# Define the filepath to save the history as an Excel file
history_filepath = "D:/iam/history.xlsx"
