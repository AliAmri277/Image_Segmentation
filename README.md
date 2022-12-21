# Image_Segmentation
 An algorithm that can automatically detect nuclei to expedite research on a wide
range of diseases, including cancer, heart disease, and rare disorders. Such tool has
the potential to significantly improve the speed at which cures are developed,
benefiting those suffering from various health conditions, including chronic
obstructive pulmonary disease, Alzheimer's, diabetes, and even the common cold.
Hence, the identification of cell nuclei is a crucial first step in many research studies
because it enables researchers to analyze the DNA contained within the nucleus,
which holds the genetic information that determines the function of each cell. By
identifying the nuclei of cells, researchers can examine how cells respond to
different treatments and gain insights into the underlying biological processes at
play. An automated AI model for identifying nuclei, offers the potential to streamline
drug testing and reduce the time for new drugs to become available to the public.

#### 1. Data Loading
        -Load dataset using OpenCV. The image was loaded in RGB format (3 channel) while the masks in grayscale (1 channel). The image is as dispalyed below.
<p align="center">
<img src="Resources/output.png" class="center"></p>

#### 2. Data Preprocessing
        -The input images and masks images is normalized to 255. Both image and masks is split using scikitlearn.model_selection method, train_test_split().
