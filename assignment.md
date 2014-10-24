<!-- Make sure that the knitr package is installed and loaded. -->
<!-- For more info on the package options see http://yihui.name/knitr/options -->

<!-- Replace below with the title of your project -->
### Pratical Machine Learning - Assignment

<!-- Enter the code required to load your data in the space below. The data will be loaded but the line of code won't show up in your write up (echo=FALSE) in order to save space-->


<!-- In the remainder of the document, add R code chunks as needed -->

### Building the model:

The first step was to load the data:


```r
original_training <- csv.get("pml-training.csv")
```

After that I took a look at the summary of the data and noticed that a lot of variables had missing values and/or were not relevant to our case, because they contained just names or dates, so I removed them from the data.


```r
summary(original_training)

remove <- c("X", "user.name", "raw.timestamp.part.1", "raw.timestamp.part.2",
            "cvtd.timestamp", "new.window", "num.window", "kurtosis.roll.belt",
            "kurtosis.picth.belt", "kurtosis.yaw.belt", "skewness.roll.belt",
            "skewness.roll.belt.1", "skewness.yaw.belt", "max.roll.belt",
            "max.picth.belt", "max.yaw.belt", "min.roll.belt", "min.pitch.belt",
            "min.yaw.belt", "amplitude.roll.belt", "amplitude.pitch.belt",
            "amplitude.yaw.belt", "var.total.accel.belt", "avg.roll.belt",
            "stddev.roll.belt", "var.roll.belt", "avg.pitch.belt", "stddev.pitch.belt",
            "var.pitch.belt", "avg.yaw.belt", "stddev.yaw.belt", "var.yaw.belt",
            "var.accel.arm", "avg.roll.arm", "stddev.roll.arm", "var.roll.arm",
            "avg.pitch.arm", "stddev.pitch.arm", "var.pitch.arm", "avg.yaw.arm",
            "stddev.yaw.arm", "var.yaw.arm", "kurtosis.roll.arm", "kurtosis.picth.arm",
            "kurtosis.yaw.arm", "skewness.roll.arm", "skewness.pitch.arm",
            "skewness.yaw.arm", "max.roll.arm", "max.picth.arm", "max.yaw.arm",
            "min.roll.arm", "min.pitch.arm", "min.yaw.arm", "amplitude.roll.arm",
            "amplitude.pitch.arm", "amplitude.yaw.arm", "kurtosis.roll.dumbbell",
            "kurtosis.picth.dumbbell", "kurtosis.yaw.dumbbell", "skewness.roll.dumbbell",
            "skewness.pitch.dumbbell", "skewness.yaw.dumbbell", "max.roll.dumbbell",
            "max.picth.dumbbell", "max.yaw.dumbbell", "min.roll.dumbbell",
            "min.pitch.dumbbell", "min.yaw.dumbbell", "amplitude.roll.dumbbell", 
            "amplitude.pitch.dumbbell", "amplitude.yaw.dumbbell", "var.accel.dumbbell",
            "avg.roll.dumbbell", "stddev.roll.dumbbell", "var.roll.dumbbell",
            "avg.pitch.dumbbell", "stddev.pitch.dumbbell", "var.pitch.dumbbell",
            "avg.yaw.dumbbell", "stddev.yaw.dumbbell", "var.yaw.dumbbell",
            "kurtosis.roll.forearm", "kurtosis.picth.forearm", "kurtosis.yaw.forearm",
            "skewness.roll.forearm", "skewness.pitch.forearm", "skewness.yaw.forearm",
            "max.roll.forearm", "max.picth.forearm", "max.yaw.forearm", "min.roll.forearm",
            "min.pitch.forearm", "min.yaw.forearm", "amplitude.roll.forearm",
            "amplitude.pitch.forearm", "amplitude.yaw.forearm", "var.accel.forearm",
            "avg.roll.forearm", "stddev.roll.forearm", "var.roll.forearm",
            "avg.pitch.forearm", "stddev.pitch.forearm", "var.pitch.forearm",
            "avg.yaw.forearm", "stddev.yaw.forearm", "var.yaw.forearm"
            )

modified_training <- original_training[, names(original_training)[names(original_training) %nin% remove]]
```

Then I checked for correlation between variables and noticed that a lot of them were highly correlated, so I decided to use preprocessing in my model.


```r
M <- abs(cor(modified_training[,-53]))
diag(M) <- 0
which(M > 0.8,arr.ind=T)
```

```
##                  row col
## yaw.belt           3   1
## total.accel.belt   4   1
## accel.belt.y       9   1
## accel.belt.z      10   1
## accel.belt.x       8   2
## magnet.belt.x     11   2
## roll.belt          1   3
## roll.belt          1   4
## accel.belt.y       9   4
## accel.belt.z      10   4
## pitch.belt         2   8
## magnet.belt.x     11   8
## roll.belt          1   9
## total.accel.belt   4   9
## accel.belt.z      10   9
## roll.belt          1  10
## total.accel.belt   4  10
## accel.belt.y       9  10
## pitch.belt         2  11
## accel.belt.x       8  11
## gyros.arm.y       19  18
## gyros.arm.x       18  19
## magnet.arm.x      24  21
## accel.arm.x       21  24
## magnet.arm.z      26  25
## magnet.arm.y      25  26
## accel.dumbbell.x  34  28
## accel.dumbbell.z  36  29
## gyros.dumbbell.z  33  31
## gyros.forearm.z   46  31
## gyros.dumbbell.x  31  33
## gyros.forearm.z   46  33
## pitch.dumbbell    28  34
## yaw.dumbbell      29  36
## gyros.forearm.z   46  45
## gyros.dumbbell.x  31  46
## gyros.dumbbell.z  33  46
## gyros.forearm.y   45  46
```

Next step was to set a seed, separate my data into a training set and a test set and build the model. I chose to preprocess the data using the PCA method and to make a 10-fold cross validation using the "train" function options.


```r
set.seed(1235)

inTraining <- createDataPartition(modified_training$classe, p = 0.75, list = FALSE)
training <- modified_training[inTraining, ]
testing <- modified_training[-inTraining, ]

modelFit <- train(training$classe ~ ., method="gbm", data=training,
                  preProcess="pca", trControl = trainControl(method = "cv",
                                                             number = 10,
                                                             repeats = 10))
```

```
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
## Warning: a condição tem comprimento > 1 e somente o primeiro elemento será usado
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 13246, 13245, 13247, 13247, 13246, 13246, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                   50      0.6       0.4    0.014        0.02    
##   1                  100      0.6       0.5    0.010        0.01    
##   1                  150      0.6       0.6    0.009        0.01    
##   2                   50      0.7       0.6    0.010        0.01    
##   2                  100      0.7       0.7    0.008        0.01    
##   2                  150      0.8       0.7    0.008        0.01    
##   3                   50      0.7       0.6    0.011        0.01    
##   3                  100      0.8       0.7    0.008        0.01    
##   3                  150      0.8       0.8    0.010        0.01    
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

The accuracy, in this case, is an optmistic measurement. To analyse better my model I then checked it against the testing data.


```r
confusionMatrix(testing$classe,predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1254   24   47   57   13
##          B  102  704   94   22   27
##          C   54   67  688   22   24
##          D   25   19  100  644   16
##          E   17   74   60   36  714
## 
## Overall Statistics
##                                         
##                Accuracy : 0.816         
##                  95% CI : (0.805, 0.827)
##     No Information Rate : 0.296         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.768         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.864    0.793    0.696    0.825    0.899
## Specificity             0.959    0.939    0.957    0.961    0.955
## Pos Pred Value          0.899    0.742    0.805    0.801    0.792
## Neg Pred Value          0.944    0.953    0.926    0.967    0.980
## Prevalence              0.296    0.181    0.202    0.159    0.162
## Detection Rate          0.256    0.144    0.140    0.131    0.146
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.911    0.866    0.826    0.893    0.927
```

With these results I expect the out of sample error to be about 20%. I guess this is not a bad number.
