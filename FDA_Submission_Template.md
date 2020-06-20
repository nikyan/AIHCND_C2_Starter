# FDA  Submission

**Your Name:**
Nikhil Narayan

**Name of your Device:**
Pneumonia Identifier

## Algorithm Description 

### 1. General Information

The Pneumona Identifier is a CAD system that assesses the whole image (x-ray) and returns an output stating whether or not Pneumonia is present in an image.

**Intended Use Statement:** 
for assisting the radiologist in the detection of Penuemonia from chest x-rays in a patient.


**Indications for Use:**

* Workflow priortization for Radiologists
* Screening x-rays for Pneumonia
* Can classify Male and Female patients between age groups of 20 and 70

In diagnostic situations, a clinician orders an imaging study because they believe that a disease may be present based on the patient's symptoms. Diagnostic imaging can be performed in emergency settings as well as non-emergency settings. The CAD system can perform the first level of analysis which can then be validated by a Radiologist. The classification done by the CAD system can be used to optimize Radiologist workflow so that positive cases are assessed first. 

**Device Limitations:**

* Requires High Computational Infrastructure (GPU, High RAM & CPU)
* The CAD system may not perform well in the presence of other diseases such as Infiltration, Edema, Atelectasis, Effusion etc.
* Requires Patient age to be between 20 and 70 years.


**Clinical Impact of Performance:**

The threshold is selected to optimize Recall so that we have more confidence when the test is negative. This is well suited for worklist priortization.


### 2. Algorithm Design and Function

The model was trained on 10 epochs. The image below presents the results from each epoch:

![GitHub Logo](/images/Epochs_Loss.png)

**DICOM Checking Steps:**
1. Get pixel data from dicom file.
2. Review important fields:
 - Disease finding
 - Patient ID, Age, Sex
 - Image Size
 3. Ensure the body part examined is 'CHEST'.

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
   1. Train: 100
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

![GitHub Logo](/images/pr_curve.png)

**Final Threshold and Explanation:**

![GitHub Logo](/images/Threshold.png)

* F1 is right metric to consider since we are dealing with imbalanced dataset and F1 considers both Precision and Recall. 
* While evaluating performance using thresholds, F1 score increases but precision starts to decrease post 0.7 threshold. 
* With the current model, 0.7 threshold is considered the final threshold.
* At 0.7, the Recall is optimized which is important from intended use perspective (worklist prioritization).
* Accuracy as a metric is misleading since the dataset is imbalanced.

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)
 * NIH data: Data_Entry_2017.csv
The dataset consists of 112120 chest x-rays with disease labels acquired from 30000 patients. The labels in the dataset were created using Natural Language Processing by text-mining disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. 
 
 * Sample NIH data: sample_labels.csv
 Sample of the NIH data used for EDA.
   
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
  
  Distribution of Pneumonia cases by Age and Gender:
  
  ![GitHub Logo](/images/Dist_Pneumonia_Age_Gender.png)
  

**Description of Training Dataset:** 

* Majority class is under sampled to have equal weightage of Pneumonia and Non-Pneumonia cases.
* The training dataset is augmented to provide more real life examples of x-ray images.
* The training batch size is kept at 100, which results in approx 22 batches and with 10 epocs, a total of 220 batches. 
* Adam optimizer is used with a learning rate of 1e-4. To prevent overfitting, EarlyStopping is used on loss function with a patience value of 10.
* Pre-trained VGG16 model is used as the base model. 3 dropout layers are added to prevent overfitting and improve generalization


**Description of Validation Dataset:** 

* The training and validation dataset are stratified by Pneumonia class i.e. there are equal number of Pneumonia and Non-Pneumonia 
cases in training and validation dataset.
* No augmentation is performed on the validation dataset.


### 5. Ground Truth

Radiologists can detect Pneumonia with high confidence from chest x-rays of patients. The labeling performed by Radiologists can be used as ground truth. Since, we are not planning to replace Radiologists, the classification performed by Radiologist can be used as gold standard.

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

WHO:
* The validation dataset should be for patients between age group of 20 and 70.
* The patients should not have any other comborbid diseases.

WHAT:
* x-ray images of chest


**Ground Truth Acquisition Methodology:**

Ground Truth: Classification performed by Radiologists.
* Identify a clinical partner who can provide data.
* Create silver standard by using multiple radiologists.
* Perform classification to validate algorithm.


**Algorithm Performance Standard:**

Taking existing research as predicate for Radiologist performance, the average F1 score is the following from 4 Radiologists:

Radiologist Avg. F1-Score: 0.387

The above F1 score for Radiologists can be used as benchmark for this algorithm and also used to evaluate whether the algorithm is at par with Radiologists in predicting Pneumonia from x-ray images.

Reference Study: https://arxiv.org/pdf/1711.05225.pdf%202017.pdf


