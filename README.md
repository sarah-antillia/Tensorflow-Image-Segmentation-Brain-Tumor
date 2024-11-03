<h2>Tensorflow-Image-Segmentation-Brain-Tumor (2024/11/03)</h2>

This is the second experiment of Image Segmentation for Brain Tumor
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/102X3nartGSh4X3LCzjSxXW33wRx-TA1P/view?usp=sharing">
LGG-MRI-Brain-Tumor-ImageMask-Dataset.zip</a>, which was derived by us from  
<a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation">
Brain MRI segmentation
</a>
<br>
<br>
Please see also our first experiment 
<a href="https://github.com/atlan-antillia/Image-Segmentation-Brain-Tumor">Image-Segmentation-Brain-Tumor
</a>
<br>
<br>
On our dataset, please refer to <a href="https://github.com/sarah-antillia/Augmentation-CDD-CESM-Brain-Tumor-Segmentation-Dataset">
Augmentation-CDD-CESM-Brain-Tumor-Segmentation-Dataset
</a>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4941_19960909_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4941_19960909_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4941_19960909_12.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_5393_19990606_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_5393_19990606_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_5393_19990606_12.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Brain-TumorSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following web site.

<b>Brain MRI segmentation</b><br>
<a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation">
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
</a>
<br><br>
<b>About Dataset</b><br>
LGG Segmentation Dataset<br>
Dataset used in:<br>
Mateusz Buda, AshirbaniSaha, Maciej A. Mazurowski "Association of genomic subtypes of<br> 
lower-grade gliomas with shape features automatically extracted by a deep learning <br>
algorithm." Computers in Biology and Medicine, 2019.<br>
and
Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi,<br> 
Katherine B. Peters, Ashirbani Saha "Radiogenomics of lower-grade glioma: <br>
algorithmically-assessed tumor shape is associated with tumor genomic subtypes <br>
and patient outcomes in a multi-institutional study with <br>
The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.<br>
<br>
This dataset contains brain MR images together with manual FLAIR abnormality <br>
segmentation masks.<br>
The images were obtained from The Cancer Imaging Archive (TCIA).<br>
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA)<br> 
lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) <br>
sequence and genomic cluster data available.<br>
Tumor genomic clusters and patient data is provided in data.csv file.<br>
<br>

<br>
<h3>
<a id="2">
2 Brain-TumorImageMask Dataset
</a>
</h3>
 If you would like to train this Brain-TumorSegmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/102X3nartGSh4X3LCzjSxXW33wRx-TA1P/view?usp=sharing">
LGG-MRI-Brain-Tumor-ImageMask-Dataset.zip</a>,
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Brain-Tumor
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Brain-Tumor Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/Brain-Tumor_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Brain-TumorTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumorand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 84 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/train_console_output_at_epoch_84.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Brain-Tumor.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/evaluate_console_output_at_epoch_84.png" width="720" height="auto">
<br><br>Image-Segmentation-CDD-CESM-Brain-Tumor

<a href="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Brain-Tumor/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0629
dice_coef,0.8953
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Brain-Tumor.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4941_19960909_17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4941_19960909_17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4941_19960909_17.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4943_20000902_16.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4943_20000902_19.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4943_20000902_19.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4943_20000902_19.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_4944_20010208_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_4944_20010208_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_4944_20010208_9.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_5393_19990606_10.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_5393_19990606_10.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_5393_19990606_10.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/images/TCGA_CS_5396_20010302_14.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test/masks/TCGA_CS_5396_20010302_14.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Brain-Tumor/mini_test_output/TCGA_CS_5396_20010302_14.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Brain MRI segmentation</b><br>
<a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation">
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
</a>
<br>
<br>
<b>2. Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images</b><br>
Ramin Ranjbarzadeh, Abbas Bagherian Kasgari, Saeid Jafarzadeh Ghoushchi, <br>
Shokofeh Anari, Maryam Naseri & Malika Bendechache <br>
<a href="https://www.nature.com/articles/s41598-021-90428-8">
https://www.nature.com/articles/s41598-021-90428-8
</a>
<br>
<br>
<b>3. Deep learning based brain tumor segmentation: a survey</b><br>
Zhihua Liu, Lei Tong, Long Chen, Zheheng Jiang, Feixiang Zhou,<br>
Qianni Zhang, Xiangrong Zhang, Yaochu Jin & Huiyu Zhou
<br>
<a href="https://link.springer.com/article/10.1007/s40747-022-00815-5">
https://link.springer.com/article/10.1007/s40747-022-00815-5
</a>
<br>
<br>
<b>4. EfficientDet-MRI-Brain-Tumor</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/EfficientDet-MRI-Brain-Tumor">
https://github.com/sarah-antillia/EfficientDet-MRI-Brain-Tumor
</a>
<br>
<br>
<b>5. Image-Segmentation-Brain-Tumor</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/atlan-antillia/Image-Segmentation-Brain-Tumor">
https://github.com/atlan-antillia/Image-Segmentation-Brain-Tumor
</a>
<br>

<br>

