# PracticeWeekProject2018Q3
A machine learning project for Q3 2018 at buildium

Once you download this repo, you will also need to populate folders in PracticeWeekProject2018Q3\CNN_Simple_Model_2_Categories\doctypes with datasets. 

There are datasets of Buildium-relevant documents that you can use to populate the `0` folder and any equally-sized number of photos will do to populate the `1` folder.
Specifically for 2-category categorical crossentropy models, you'll need to have your folder labels be integers like this instead of the usual strings due to a need to use 
keras' to_categorical function on your y data instead of LabelBinarizer (LB will not properly return a vector, as our model requires, and to_categorical only understands integers)

To train this model, run this command, adjusting for the correct path on your machine for your user:
`python .\train_doctype_2category.py --dataset 'C:\Users\alla.hoffman\Documents\tensorflow\PracticeWeekProject2018Q3\CNN_Simple_Model_2_Categories\doctypes'  --model ../model --label-bin ../model --plot 'C:\Users\alla.hoffman\Documents\tensorflow\PracticeWeekProject2018Q3\CNN_Simple_Model_2_Categories\docImages'`

NB that if you are working on a windows machine, this command may only work from the command prompt within an issue of VisualStudio Code - or this may be a result of idiosyncratic Python installations on some local machines.