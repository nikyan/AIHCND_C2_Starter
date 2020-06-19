# FDA  Submission

**Your Name:**
Nikhil Narayan

**Name of your Device:**
Pneumonia Identifier

## Algorithm Description 

### 1. General Information

The Algorithm designed is a classification algorithm performing binary classification that assesses the whole image (x-ray) and returns an output stating whether or not Pneumonia is present in an image.

**Intended Use Statement:** 

In diagnostic situations, a clinician orders an imaging study because they believe that a disease may be present based on the patient's symptoms. Diagnostic imaging can be performed in emergency settings as well as non-emergency settings.
The Pneumonia Identifier 

**Indications for Use:**




**Device Limitations:**



**Clinical Impact of Performance:**

### 2. Algorithm Design and Function

The model was trained on 10 epochs. The image below presents the results from each epoch:

![GitHub Logo](/images/Epochs_Loss.png)

**DICOM Checking Steps:**
1. Get pixel data from dicom file.
2. Review important fields:
 - Disease finding
 - Patient ID, Age, Sex
 - Image Size

**Preprocessing Steps:**
1. Resize image to match VGG16 input requirement
2. Standardize image by subtracting mean of training data and dividing by standard deviation of training data.


**CNN Architecture:**
Pneumonia Screener uses VGG16 Convolutional Neuro Net as base model with additional layers for fine-tuning. The additional layers can be seen in the image below: 

![GitHub Logo](/images/Model_Seq.png)

### 3. Algorithm Training

**Parameters:**

* Types of augmentation used during training:

   1. horizontal_flip = True
   2. height_shift_range = 0.05
   3. width_shift_range=0.1
   4. rotation_range=5
   5. shear_range = 0.1
   6. fill_mode = 'reflect'
   7. zoom_range=0.15
 
* Batch size
   1. Train: 32
   2. Test: 1024
 
* Optimizer learning rate: Adam(lr=1e-4)

* Layers of pre-existing architecture that were frozen: 17 layers
  
* Layers of pre-existing architecture that were fine-tuned
    1. block5_conv3
    2. block5_pool

* Layers added to pre-existing architecture
    1. flatten_3 (Flatten)    
    2. dropout_7 (Dropout)         
    3. dense_9 (Dense)            
    4. dropout_8 (Dropout)         
    5. dense_10 (Dense)            
    6. dropout_9 (Dropout)          
    7. dense_11 (Dense)            
    8. dense_12 (Dense)
 
* Training Loss and Accuracy on Dataset

![GitHub Logo](/images/history.png)


* PR Curve


**Final Threshold and Explanation:**

![GitHub Logo](/images/Threshold.png)

F1 score increases till 0.6 and then stabilizes while Precision starts to go up and suddenly jumps around 0.8 suggesting that the model is majority class.
Accuracy here is misleading since the dataset is imbalanced.

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)
 * NIH data: Data_Entry_2017.csv
 * PNG Images: X-ray images
  112120 images and labels in csv file.
 
 * Sample NIH data: sample_labels.csv
   Sample data used for EDA.
   
**Data Exploration**

 * Demographics
 
 Distribution of Patients by Age:
 
 ![GitHub Logo](/images/Dist_Patient_Age.png)
 
 Distribution of Patients by Gender:
 
 ![GitHub Logo](/images/Dist_Patient_Gender.png)
 
 
 Distribution of X-rays by View Position:

 ![GitHub Logo](/images/Dist_View_Position.png)
 
 * Diseases
 
  Distribution of all diseases in the dataset:
  
  ![GitHub Logo](/images/Dist_all_images.png)
  
  Count of Pneumonia cases in the dataset:
  
  ![GitHub Logo](/images/Dist_Pneumonia_Cases.png)
  

**Description of Training Dataset:** 

Majority class is under sampled to have equal weightage of Pneumonia and Non-Pneumonia cases.

Training data shape: (2290, 29)



**Description of Validation Dataset:** 

The training and validation dataset are stratified by Pneumonia class i.e. there are equal number of Pneumonia and Non-Pneumonia 
cases in training and validation dataset.

Validation data shape: (22424, 29)

### 5. Ground Truth

Since we are using this algorithm for screening, the labeling by Radiologist can be used as ground truth.


### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**
You need to gather the ground truth that can be used to compare the model output tested on the FDA validation set. The choice of your ground truth method ties back to your intended use statement. Depending on the intended use of the algorithm, the ground truth can be very different.

**Ground Truth Acquisition Methodology:**

Validation Plan
Performance standard
For your validation plan, you need evidence to support your reasoning. As a result, you need a performance standard. This step usually involves a lot of literature searching.
Depending on the use case for your algorithm, part of your validation plan may need to include assessing how fast your algorithm can read a study.


**Algorithm Performance Standard:**
