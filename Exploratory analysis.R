library(caret)
library(Hmisc)
library(ggplot2)

original_training <- csv.get("pml-training.csv")
#original_testing <- csv.get("pml-testing.csv")

summary(original_training)

# Medidas a desconsiderar
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

# Monta matriz de treinamento
modified_training <- original_training[, names(original_training)[names(original_training) %nin% remove]]
#testing <- original_testing[, names(original_testing)[names(original_testing) %nin% remove]]

# Verifica correlação entre variáveis
M <- abs(cor(modified_training[,-53]))
diag(M) <- 0
which(M > 0.8,arr.ind=T)

# Grupos: belt, arm, dumbbell, forearm

# Por ter muita correlação, farei preprocessamento
# Escolho o seed
set.seed(1235)

# Separo em treino e teste
inTraining <- createDataPartition(modified_training$classe, p = 0.75, list = FALSE)
training <- modified_training[inTraining, ]
testing <- modified_training[-inTraining, ]

# Crio o modelo
modelFit <- train(training$classe ~ ., method="gbm", data=training,
                  preProcess="pca", trControl = trainControl(method = "cv",
                                                             number = 10,
                                                             repeats = 10))

# Teste
confusionMatrix(testing$classe,predict(modelFit,testing))