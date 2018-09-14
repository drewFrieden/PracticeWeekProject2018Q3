# PracticeWeekProject2018Q3
A machine learning project for Q3 2018 at buildium

Folders:

ImageDownload - A project for scraping images from google search.
Notebooks - Some example utility notebooks as well as our final presentation notebook.
CNN_Model_Multi_Layer_2_Category - A Convultional Neural Network project that uses two categories of images to create a model.
CNN_Simple_Model_2_Categories - A simple dense layer machine learning model that uses 2 categories.
CNN_Simple_Model_Binary - Same as CNN_Simple_Model_2_Categories but it uses a different loss algorithm: binary_crossentropy vs categorical_crossentropy.
API_CNN_Simple - A simple flask endpoint that uses the CNN_Simple_Model_2_Categories example as its model.

Example run for CNN_Model_Multi_Layer_2_Category:

python .\train_vgg.py --dataset '..\PracticeWeekProject2018Q3\CNN_Model_Multi_Layer_2_Category\Test_Data' --model "..\PracticeWeekProject2018Q3\CNN_Model_Multi_Layer_2_Category\Model" --plot
'..\PracticeWeekProject2018Q3\CNN_Model_Multi_Layer_2_Category\Output'

Example request for API:

curl -X POST -F image=@1099.jpg "http://localhost:5000/predict"