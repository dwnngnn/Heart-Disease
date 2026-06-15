# Thư viện
library(dplyr)
library(caret)
library(ranger)
library(pROC)

# 5.1. Chia tập dữ liệu
heart_data <- read.csv(file.path("heart_disease_dataset_clean.csv"), stringsAsFactors = FALSE)
if ("target" %in% names(heart_data)) {
  names(heart_data)[names(heart_data) == "target"] <- "heart_disease"
}
heart_data$heart_disease <- factor(heart_data$heart_disease, levels = c(0, 1))

set.seed(42)
train_index <- createDataPartition(heart_data$heart_disease, p = 0.7, list = FALSE)
training_set <- heart_data[train_index, ]
testing_set <- heart_data[-train_index, ]

# Định nghĩa hàm đánh giá chung
evaluate_model <- function(actual, predicted_class, predicted_prob = NULL) {
  predicted_class <- factor(predicted_class, levels = levels(actual))
  accuracy <- mean(predicted_class == actual)
  precision <- posPredValue(predicted_class, actual, positive = "1")
  recall <- sensitivity(predicted_class, actual, positive = "1")
  spec <- specificity(predicted_class, actual, negative = "0")
  f1_score <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  auc_val <- NA
  if (!is.null(predicted_prob)) {
    # ensure numeric vector of probabilities for the positive class ("1")
    roc_obj <- try(pROC::roc(response = actual, predictor = predicted_prob, levels = c("0", "1"), direction = "<"), silent = TRUE)
    if (!inherits(roc_obj, "try-error")) {
      auc_val <- as.numeric(pROC::auc(roc_obj))
    }
  }
  data.frame(
    Accuracy = round(accuracy, 4),
    Precision = round(precision, 4),    
    Recall = round(recall, 4),
    Specificity = round(spec, 4),
    AUC = ifelse(is.na(auc_val), NA, round(auc_val, 4)),
    F1 = round(f1_score, 4)
  )
}

results <- tibble::tibble(
  Model = character(), 
  Accuracy = numeric(), 
  Precision = numeric(), 
  Recall = numeric(), 
  Specificity = numeric(), 
  AUC = numeric(), 
  F1 = numeric()
)

# 5.2. Xây dựng mô hình Logistic Regression
model_lr <- glm(heart_disease ~ ., data = training_set, family = binomial)
pred_prob <- predict(model_lr, newdata = testing_set, type = "response")
predicted_lr <- factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0, 1))
metrics_lr <- evaluate_model(testing_set$heart_disease, predicted_lr, predicted_prob = pred_prob)
results <- bind_rows(results, tibble::tibble(Model = "Logistic Regression", metrics_lr))

cat("5.2 Logistic Regression\n")
print(metrics_lr)
cat("\n")

# 5.3. Xây dựng mô hình Random Forest
model_rfc <- ranger(
  formula = heart_disease ~ .,
  data = training_set,
  num.trees = 150,
  max.depth = 5,
  min.node.size = 5,
  probability = TRUE,
  seed = 42
)
pred_rfc <- predict(model_rfc, data = testing_set)$predictions
# when probability = TRUE, ranger returns a matrix/data.frame with columns for each class
if (is.matrix(pred_rfc) || is.data.frame(pred_rfc)) {
  predicted_rfc_prob <- as.numeric(pred_rfc[, "1"])
  predicted_rfc <- factor(ifelse(predicted_rfc_prob > 0.5, 1, 0), levels = c(0, 1))
} else {
  # fallback if predictions are class labels
  predicted_rfc <- factor(pred_rfc, levels = c(0, 1))
  predicted_rfc_prob <- NULL
}

metrics_rfc <- evaluate_model(testing_set$heart_disease, predicted_rfc, predicted_prob = predicted_rfc_prob)
results <- bind_rows(results, tibble::tibble(Model = "Random Forest", metrics_rfc))

cat("5.3 Random Forest\n")
print(metrics_rfc)
cat("\n")

# 5.4. Xây dựng mô hình KNN
model_knn <- knn3(heart_disease ~ ., data = training_set, k = 7)
pred_knn_prob_mat <- predict(model_knn, newdata = testing_set, type = "prob")
if (!is.null(pred_knn_prob_mat)) {
  predicted_knn_prob <- as.numeric(pred_knn_prob_mat[, "1"])
  predicted_knn <- factor(ifelse(predicted_knn_prob > 0.5, 1, 0), levels = c(0, 1))
} else {
  predicted_knn <- predict(model_knn, newdata = testing_set, type = "class")
  predicted_knn_prob <- NULL
}

metrics_knn <- evaluate_model(testing_set$heart_disease, predicted_knn, predicted_prob = predicted_knn_prob)
results <- bind_rows(results, tibble::tibble(Model = "KNN", metrics_knn))

cat("5.4 KNN\n")
print(metrics_knn)
cat("\n")

# 5.5. So sánh các mô hình
cat("5.5 So sánh các mô hình\n")
print(results)

# 5.6. In số liệu hệ số / tầm quan trọng cho từng mô hình (chỉ in bảng/ma trận)

# Hàm in cho Logistic Regression (chỉ in bảng hệ số và OR với CI)
interpret_logistic <- function(model) {
  s <- summary(model)
  coefs <- s$coefficients
  est <- coef(model)
  se <- sqrt(diag(vcov(model)))
  z <- qnorm(0.975)
  or <- exp(est)
  or_low <- exp(est - z * se)
  or_high <- exp(est + z * se)
  df <- data.frame(
    Variable = names(est),
    Coefficient = round(est, 4),
    OR = round(or, 4),
    OR_95CI = paste0(round(or_low,4), " - ", round(or_high,4)),
    P_value = ifelse(rownames(coefs) %in% rownames(coefs), round(coefs[,4], 4), NA)
  )
  print(df)
}

# Hàm in tầm quan trọng biến cho Random Forest (refit để lấy importance)
interpret_ranger <- function(training_data, num.trees = 350, max.depth = 5, seed = 42) {
  rf_imp <- ranger(
    formula = heart_disease ~ .,
    data = training_data,
    num.trees = num.trees,
    max.depth = max.depth,
    probability = TRUE,
    importance = "permutation",
    seed = seed
  )
  imp <- rf_imp$variable.importance
  imp_df <- data.frame(Variable = names(imp), Importance = as.numeric(imp))
  imp_df <- imp_df[order(-imp_df$Importance), ]
  print(head(imp_df, 20))
}

# Hàm permutation importance cho KNN (in bảng giá trị importance)
interpret_knn_permutation <- function(model, testing_data, response_name = "heart_disease") {
  requireNamespace("pROC", quietly = TRUE)
  predictors <- setdiff(names(testing_data), response_name)
  # baseline
  prob_mat <- try(predict(model, newdata = testing_data, type = "prob"), silent = TRUE)
  if (!inherits(prob_mat, "try-error") && !is.null(prob_mat)) {
    baseline_prob <- as.numeric(prob_mat[, "1"])
    roc_obj <- try(pROC::roc(response = testing_data[[response_name]], predictor = baseline_prob, levels = c("0","1"), direction = "<"), silent = TRUE)
    baseline_auc <- if (!inherits(roc_obj, "try-error")) as.numeric(pROC::auc(roc_obj)) else NA
  } else {
    pred_class <- try(predict(model, newdata = testing_data, type = "class"), silent = TRUE)
    baseline_acc <- if (!inherits(pred_class, "try-error")) mean(pred_class == testing_data[[response_name]]) else NA
    baseline_auc <- NA
  }
  imp_values <- numeric(length(predictors))
  names(imp_values) <- predictors
  for (v in predictors) {
    perm_data <- testing_data
    perm_data[[v]] <- sample(perm_data[[v]])
    prob_mat_p <- try(predict(model, newdata = perm_data, type = "prob"), silent = TRUE)
    if (!inherits(prob_mat_p, "try-error") && !is.null(prob_mat_p) && !is.na(baseline_auc)) {
      perm_prob <- as.numeric(prob_mat_p[, "1"])
      roc_p <- try(pROC::roc(response = perm_data[[response_name]], predictor = perm_prob, levels = c("0","1"), direction = "<"), silent = TRUE)
      perm_auc <- if (!inherits(roc_p, "try-error")) as.numeric(pROC::auc(roc_p)) else NA
      imp_values[v] <- ifelse(is.na(baseline_auc) || is.na(perm_auc), NA, baseline_auc - perm_auc)
    } else {
      pred_class_p <- try(predict(model, newdata = perm_data, type = "class"), silent = TRUE)
      acc_p <- if (!inherits(pred_class_p, "try-error")) mean(pred_class_p == perm_data[[response_name]]) else NA
      imp_values[v] <- ifelse(is.na(baseline_acc) || is.na(acc_p), NA, baseline_acc - acc_p)
    }
  }
  imp_df <- data.frame(Variable = names(imp_values), Importance = imp_values)
  imp_df <- imp_df[order(-imp_df$Importance), ]
  print(head(imp_df, 20))
}

# Gọi các hàm diễn giải (chỉ in số liệu)
interpret_logistic(model_lr)
interpret_ranger(training_set)
interpret_knn_permutation(model_knn, testing_set)
