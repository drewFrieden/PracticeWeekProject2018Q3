# PracticeWeekProject2018Q3
A machine learning project for Q3 2018 at buildium

# PracticeWeekProject2018Q3
A machine learning project for Q3 2018 at buildium

Once you download this repo, you will also need to populate folders in PracticeWeekProject2018Q3\CNN_Simple_Model_Binary\doctypes with datasets. 

There are datasets of Buildium-relevant documents that you can use to populate the `forms` folder and any equally-sized number of photos will do to populate the `images` folder.

To train this model, run this command, adjusting for the correct path on your machine for your user:
`python .\train_doctype.py --dataset 'C:\Users\alla.hoffman\Documents\tensorflow\PracticeWeekProject2018Q3\CNN_Simple_Model_Binary\doctypes'  --model ../model --label-bin ../model --plot 'C:\Users\alla.hoffman\Documents\tensorflow\PracticeWeekProject2018Q3\CNN_Simple_Model_Binary\docImages'`

NB that if you are working on a windows machine, this command may only work from the command prompt within an issue of VisualStudio Code - or this may be a result of idiosyncratic Python installations on some local machines.