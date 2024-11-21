# Alzheimer's Detection System BaseLine Model
## Dataset
### We have used the data of 77 patients from 3 classes (AD,CN,MCI). The downloaded images were in .nii format which have been conveted to .png format, with of the 3d images producing multiple slices.
![Figure_1](https://github.com/user-attachments/assets/5de432d2-ac03-48e0-b8df-7e7ad2a6ade4)
![Figure_2](https://github.com/user-attachments/assets/160ccf13-3a77-4e30-b6ea-af6a4db5c6a1)

![image](https://github.com/user-attachments/assets/76851daf-a3e7-4206-9b31-abf0eb38d335)

### After converting the images to png (nii_to_png.py), we ended up having around 14000 png images. We then trained them on the ResNet model and the results have been shown below.
### model_resnet.py need to be run after getting the train_test_split from the link below (Important: The folder should not be placed within the code folder).
### https://drive.google.com/drive/folders/1F7GTDNc4zQTfdw-9lFa2PIJu0znqsb3m?usp=drive_link
## Results
![image](https://github.com/user-attachments/assets/fc2654a8-494f-4c8e-ae58-f9b4640c516c)
