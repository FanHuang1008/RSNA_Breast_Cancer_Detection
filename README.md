# Solution for RSNA Screening Mammography Breast Cancer Detection

This repository contians the solution for the Kaggle competition: RSNA Screening Mammography Breast Cancer Detection (https://www.kaggle.com/competitions/rsna-breast-cancer-detection). The goal of this competition is to identify breast cancer with screening mammograms. In this competition, I learned the basic knowledge of medical image analysis, how to stablize the training procedure when the dataset is highly unbalanced (positive:negative = 1:49), and how to utilize Nvidia Data Loading Library (DALI) to speed up the DICOM image decoding procedure.

## Dependencies

- python==3.8
- pytorch==1.13.1(+cu116)
- torchvision==0.14.1(+cu116)
- pydicom==2.4.4
- dicomsdl==0.109.3
- matplotlib==3.7.5
- nvidia-dali-nightly-cuda110==1.39.0.dev20240527
- numpy==1.24.4
- pandas==2.0.3
- notebook==7.1.2
- opencv-python==4.9.0.80


## Preprocessing

#### Decoding DICOM with Nvidia DALI

When decoding DICOM images, we need to check whether the transfer syntax UID equals to "1.2.840.10008.1.2.4.90", which represents JPEG 2000 Image Compression (Lossless Only). If it does, we can use pydicom to find the jpeg2000 header and use NumPy to convert the buffer into array. If it doesn't, since decoding lossless JPEG is not supported by nvjpeg on GPU, we simply use dicomsdl to retrieve the pixel data. Therefore, we need to create one iterator for JPEG2000 and another for lossless JPEG.

The next step is to build two pipelines that take the iterators we just created as input. Inside the pipeline, images will be decoded with both CPU and GPU, resized with GPU, and casted into INT16. If we need to decode only a few images, we can simply build and run the pipelines then process their outputs. If a massive amount of data needs to be decoded, we can use a customized DALIGenericIterator that decodes and process the images in batch.

#### Image Processing

After decoding and resizing the DICOM images with Nvidia DALI, a YOLOv5 model is used to generate ROI. Since the YOLO model was trained on images without applying windowing, its input is also image without applying windowing. In case that YOLOv5 fails, a CV2 fallback is used to detect ROI. If the CV2 fallback also fails, the ROI is simply the whole image.

Before cropping the decoded images with the ROI, we will apply windowing to them to adjust their brightness and contrast. The cropped images are then resized into a predefined size (2048 x 1024). If the height or width of an image cannot reach the frame, padding will be added to the image. Lastly, the resized images are saved as png files.  

## Training

The training data is splitted into 4 folds and each fold is saved as a pickle file. The number of epochs and batch size for each fold equal 50 and 16, respectively. Initial leaning rate is set to 1e-5 and cosine annealing learning rate scheduler is used with T_max=4, so the learning rate decreases to 0 every 4 batches. This competition is basically a binary image classification challenge, so Binary Cross Entropy loss with mean reduction is used as the criterion. Adam, which can be thought of as a combination of Momemtum and AdaGrad, is used as the optimizer. 

To prevent the model from only predicitng values that are close to 1, soft positive lebel trick is utilized during training. In more details, all positive labels are decreased to 0.8, making the model become skepital about the correctness of positive samples. Since the dataset is highly unbalanced, a custom sampler that ensures at least 1 positive sample appears in each batch is implemented to stabalize the whole training procedure. Due to its size (#Params = 9.2M) and performance (ImageNet Top-1 Accuracy = 80.1%), EfficientNet-B2 was chosen as the training model.

To run the training script, execute the following command:
```
$ python3 train.py --seed=42 --label-smooth=True --resume="/path/to/ckpt"
```

## Validation

Since ture test data from the competition is hidden, Out-of-Fold (OOF) validation is used to evaluate the performance of the trained models. Specifically, predictions from each fold are aggregated into 1 set of predictions. We can then compute the validation loss, plot the ROC curve, precision-recall curve, sensitivity-specificity curve, and confusion matrix in one go. 

To validate the trained models in each fold, execute the following command:

```
$ python3 val.py
```

## Inference

The inference procedure is similar to that during training. The main difference is that the inference results are the average of the predctions from models in each fold.    

Execute the following command for inference:

```
$ python3 inference.py --threshold=0.01
```

## References
1st place solution write-up: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449

Decode jpeg2000 dicom with DALI: https://www.kaggle.com/code/tivfrvqhs5/decode-jpeg2000-dicom-with-dali 

Yolov5 ROI Batch DALI Preprocessing Pipeline: https://www.kaggle.com/code/outwrest/yolov5-roi-batch-dali-preprocessing-pipeline 

Elegant end2end DALI infer in Pytorch style: https://www.kaggle.com/code/forcewithme/rsna-elegant-end2end-dali-infer-in-pytorch-style
