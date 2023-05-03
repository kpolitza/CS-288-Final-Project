load("C:/Users/polit/OneDrive/Desktop/CS 288/DS0001/38062-0001-Data.rda")
library(randomForest)
library(gbm)
library(boot)
library(caret)
RNGversion('3.5.3')

# FINANCIAL # 
#(financial, categorical) 
da38062.0001$IMP_Q3MEDICALDEBT_D # Yes/No medical debt 
da38062.0001$IMP_Q3EDUCDEBT_D # Education-related debt 

# (financial, quantitative) 
da38062.0001$Q5C20X_TC # Spending on recreation, entertainment, or fitness 
da38062.0001$QA9X_TC # Monthly rent

# NON-FINANCIAL #
#(non-financial, quantitative)
da38062.0001$Q6B27X_TC # Minutes to get from home to work
da38062.0001$IMP_AGE_TC # Age

# (non-financial, categorical)
da38062.0001$Q2C2_11 # Subjective feeling of parks in their neighborhood
da38062.0001$QI4 # Access to internet on cellphone, tablet, or mobile device?
da38062.0001$IMP_PFOOD1 # Best describes food eaten in household
da38062.0001$IMP_HEALTH # Overall Health self-reported

###### Complete Cases ######## 
# Extracts the observations in the dataset for which all fin/non-fin variables are present 
complete_cases = function(dataset, response) { 
  financial = c('IMP_Q3MEDICALDEBT_D', 'IMP_Q3EDUCDEBT_D', 'Q5C20X_TC', 'QA9X_TC') 
  non_financial = c('Q6B27X_TC', 'IMP_AGE_TC', 'Q2C2_11', 'QI4', 'IMP_PFOOD1', 'IMP_HEALTH') 
  
  complete = complete.cases(dataset[c(financial, non_financial, response)]) 
  return(dataset[complete, c(financial, non_financial, response)]) 
}
subsetted_data_reg = complete_cases(da38062.0001, response=c('QD6')) 
subsetted_data_log = complete_cases(da38062.0001, response=c('QD6')) 
subsetted_data_log$QD6_bin = as.integer(subsetted_data_reg$QD6 > 5) 
subsetted_data_log = subsetted_data_log[, names(subsetted_data_log) != 'QD6']

subsetted_data_log 
subsetted_data_reg

###### Convert To Classes ########
binary_vars = c('IMP_Q3MEDICALDEBT_D', 'IMP_Q3EDUCDEBT_D', 'QI4')
classes_to_int = function(dataset, binary_vars) {
  for(var in names(dataset)) {
    dataset[var][,1] = as.numeric(dataset[var][,1])
    # Convert Yes/No values to 0/1 values
    if(var %in% binary_vars) {
      dataset[var] = rapply(dataset[var], function(x) ifelse(x==2, 0, x), how="replace")
    }
  }
  return(dataset)
}

numeric_data = classes_to_int(subsetted_data_reg, binary_vars)
numeric_data_log = classes_to_int(subsetted_data_log, binary_vars)
numeric_data
numeric_data_log

#REGRESSION -----
set.seed(1)
#data split
train = sample(1:nrow(numeric_data), 460)
test = numeric_data[-train, "QD6"]

#linear regression model --- all variables
mod = lm(QD6 ~ ., data = numeric_data, subset = train)

yhat = predict(mod, newdata = numeric_data[-train,])
testMSE = mean((yhat - test)^2)
testMSE
yhat2 = predict(mod, newdata = numeric_data[train,])
trainMSE = mean((yhat2 - numeric_data[train,"QD6"])^2)
trainMSE

#bootstrapping
boot.fn = function(data, index){
  return(coef(lm(QD6 ~ ., data=data , subset=index)))
}
boot(numeric_data, boot.fn, 1000)
summary(mod)

#10-fold CV
set.seed(1)
glm.fit = glm(QD6 ~ ., data = numeric_data)
cv.error = cv.glm(numeric_data, glm.fit, K=10)
cv.error$delta

#linear regression model --- financial variables
mod = lm(QD6 ~ IMP_Q3MEDICALDEBT_D + IMP_Q3EDUCDEBT_D + Q5C20X_TC + QA9X_TC, 
         data = numeric_data, 
         subset = train)
yhat = predict(mod, newdata = numeric_data[-train,])
testMSE = mean((yhat - test)^2)
testMSE
yhat2 = predict(mod, newdata = numeric_data[train,])
trainMSE = mean((yhat2 - numeric_data[train,"QD6"])^2)
trainMSE

#linear regression --- reduced model w/ k = 10 cv
set.seed(1)
glm.fit = glm(QD6 ~ IMP_PFOOD1 + IMP_HEALTH + IMP_Q3MEDICALDEBT_D, data = numeric_data)
cv.error = cv.glm(numeric_data, glm.fit, K=10)
cv.error$delta

#random forest --- regression
rf.mod = randomForest(QD6 ~ ., data = numeric_data, subset = train, 
                      mtry = 4, ntrees = 500, importance = TRUE)
yhat = predict(rf.mod, newdata = numeric_data[-train,])
testMSE = mean((yhat - test)^2)
testMSE
yhat2 = predict(rf.mod, newdata = numeric_data[train,])
trainMSE = mean((yhat2 - numeric_data[train,"QD6"])^2)
trainMSE

#boosting --- regression
set.seed(1)
boost.mod = gbm(QD6 ~ ., data = numeric_data[train,],
                     n.trees=5000, interaction.depth=3,
                     shrinkage = 0.01, cv.folds = 5)
yhat.boost = predict(boost.mod, newdata = numeric_data[-train ,],
                     n.trees=5000)

testMSE = mean((yhat.boost - test)^2)
testMSE
yhat2.boost = predict(boost.mod, newdata = numeric_data[train,],
                      n.trees=5000)
trainMSE = mean((yhat2.boost - numeric_data[train,"QD6"]))
trainMSE

#CLASSIFICATION -----
#logistic regression ---
set.seed(1)
test = numeric_data_log[-train, "QD6_bin"]
log.reg = glm(QD6_bin ~ ., data = numeric_data_log, subset = train, family = "binomial")
yhat.log = predict(log.reg, type = "response", newdata = numeric_data_log[-train,])
glm.pred = rep("0", 197)
glm.pred[yhat.log > .5]= "1"
table(glm.pred, test)
accuracy = (158 + 8) / (8 + 4 + 27 + 158)
accuracy
#10-fold cv
set.seed(1)
data.copy = numeric_data_log
data.copy$QD6_bin = factor(data.copy$QD6_bin, levels = c("1", "0"))
cvResults = train(QD6_bin ~ ., data = data.copy, 
                   method = "glm", 
                   family = "binomial",
                   trControl = trainControl(method = "cv", number = 10))
summary(cvResults)
cvResults

#random forest --- classification
set.seed(1)
rf.mod.class = randomForest(QD6_bin ~ ., data = data.copy[train,], 
                    mtry = 4, ntrees = 500, importance = TRUE)
test = data.copy[-train, "QD6_bin"]
yhat = predict(rf.mod.class, type = "response", newdata = data.copy[-train,])
yhat_num <- as.numeric(as.character(yhat))
rf.pred = rep("0", 197)
rf.pred[yhat_num > .5]= "1"
table(rf.pred, test)
accuracy = (156 + 3) / (6 + 3 + 156 + 32)
accuracy

#boosting --- classification
set.seed(1)
test = data.copy[-train, "QD6_bin"]
boost.mod.class = gbm(QD6_bin ~ ., data = numeric_data_log[train,],
                n.trees=5000, interaction.depth=3,
                shrinkage = 0.01, cv.folds = 5)
yhat.boost = predict(boost.mod.class, type = "response", newdata = numeric_data_log[-train ,],
                     n.trees=5000)
boost.yhat_num <- as.numeric(as.character(yhat.boost))
boost.pred = rep("0", 197)
boost.pred[boost.yhat_num > .5]= "1"
table(boost.pred, test)
accuracy = (146 + 6) / (16 + 6 + 29 + 146)
accuracy
