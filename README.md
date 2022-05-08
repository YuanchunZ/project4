# project4
A car sign data set was selected, and the pictures were stored in the data folder. Through machine learning and feature extraction, the original feature set was compared, the feature set was reduced to 3 dimensions, and the feature set was reduced to 2 dimensions. Finally, the data set features were displayed on the coordinate axis through visualization.

Dimension reduction method（PCA）

Examples of selected datasets are shown below：A total of 16 categories of data sets were found.
![car-logo-data](https://github.com/YuanchunZ/project4/blob/main/IMG_3553.JPG)

The extracted original features were put into the train_feature.npy file, and a total of 34 category features were trained. Features reduced to 2 dimensions were put into train_Feature_reduction2.npy, and features reduced to 3 dimensions were put into train_Feature_reduction3.npy.The code is get_feature.py.

The feature is visualized to the coordinate axis. As shown in the following figure, a total of 34 different colors represent 34 categories：
![2-dimensions](https://github.com/YuanchunZ/project4/blob/main/IMG_3552.JPG)

![2-dimensions](https://github.com/YuanchunZ/project4/blob/main/IMG_3551.JPG)

Extracting features and reducing dimensions to two and three dimensions for training, the accuracy of xGBoost training is 1：

![train](https://github.com/YuanchunZ/project4/blob/main/IMG_3550.PNG)
