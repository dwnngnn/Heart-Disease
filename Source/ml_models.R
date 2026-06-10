# ============================================================================
# ML Models Module - Dự Báo Bệnh Tim
# ============================================================================
# 5.2. Logistic Regression
# 5.3. Random Forest  
# 5.5. KNN
# ============================================================================

library(caret)
library(randomForest)
library(ggplot2)
library(rlang)

# 5.2. LOGISTIC REGRESSION
build_logistic_regression <- function(train_data, predictors, target_col, cv_folds = 5, seed = 42) {
  set.seed(seed)
  
  ctrl <- trainControl(
    method = "cv", 
    number = cv_folds, 
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  formula <- as.formula(paste(target_col, "~", paste(predictors, collapse = "+")))
  
  train(
    formula,
    data = train_data,
    method = "glm",
    family = "binomial",
    trControl = ctrl,
    metric = "ROC"
  )
}

# 5.3. RANDOM FOREST
build_random_forest <- function(train_data, predictors, target_col, cv_folds = 5, seed = 42) {
  set.seed(seed)
  
  ctrl <- trainControl(
    method = "cv", 
    number = cv_folds,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  formula <- as.formula(paste(target_col, "~", paste(predictors, collapse = "+")))
  
  train(
    formula,
    data = train_data,
    method = "rf",
    trControl = ctrl,
    metric = "ROC",
    tuneGrid = data.frame(mtry = c(3, 5, 7))
  )
}

# 5.5. KNN
build_knn <- function(train_data, predictors, target_col, cv_folds = 5, seed = 42) {
  set.seed(seed)
  
  ctrl <- trainControl(
    method = "cv", 
    number = cv_folds,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  formula <- as.formula(paste(target_col, "~", paste(predictors, collapse = "+")))
  
  train(
    formula,
    data = train_data,
    method = "knn",
    trControl = ctrl,
    metric = "ROC",
    preProcess = c("center", "scale"),
    tuneGrid = data.frame(k = seq(3, 21, by = 2))
  )
}

# ĐÁNH GIÁ
evaluate_model <- function(model, test_data, test_target, positive_class = "yes") {
  predictions <- predict(model, test_data)
  confusionMatrix(predictions, test_target, positive = positive_class)
}

compute_metrics <- function(cm) {
  acc <- cm$overall["Accuracy"]
  prec <- cm$byClass["Pos Pred Value"]
  rec <- cm$byClass["Sensitivity"]
  f1 <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
  c(Accuracy = acc, Precision = prec, Recall = rec, F1 = f1)
}

compare_models <- function(models_list, test_data, test_target, model_names = names(models_list)) {
  results <- list()
  cms <- list()
  
  for (i in seq_along(models_list)) {
    cm <- evaluate_model(models_list[[i]], test_data, test_target)
    metrics <- compute_metrics(cm)
    results[[model_names[i]]] <- metrics
    cms[[model_names[i]]] <- cm
  }
  
  comparison_df <- as.data.frame(t(do.call(rbind, results)))
  list(metrics = comparison_df, confusion_matrices = cms)
}

plot_logistic_regression <- function(model, test_data, test_target) {
  probs <- predict(model, test_data, type = "prob")
  
  df_plot <- data.frame(
    Probability = probs[, 2],
    Actual = ifelse(test_target == "yes", 1, 0)
  )
  
  ggplot(df_plot, aes(x = Probability, fill = factor(Actual))) +
    geom_histogram(alpha = 0.6, bins = 30) +
    labs(title = "Logistic Regression - Phân Bố Xác Suất",
         x = "Xác Suất Mắc Bệnh",
         y = "Tần Số",
         fill = "Thực Tế") +
    scale_fill_manual(values = c("0" = "blue", "1" = "red"),
                      labels = c("0" = "Không Bệnh", "1" = "Mắc Bệnh")) +
    theme_minimal()
}

plot_random_forest <- function(model, top_n = 15) {
  importance <- varImp(model)
  imp_df <- as.data.frame(importance$importance)
  imp_df$Feature <- rownames(imp_df)
  imp_df <- imp_df[order(-imp_df$Overall), ]
  imp_df <- head(imp_df, top_n)
  
  ggplot(imp_df, aes(x = reorder(Feature, Overall), y = Overall)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest - Độ Quan Trọng Đặc Trưng",
         x = "Đặc Trưng",
         y = "Độ Quan Trọng") +
    theme_minimal()
}

plot_knn_boundary <- function(model, train_data, target_col, feature1, feature2) {
  h <- 0.02
  x1_min <- min(train_data[[feature1]], na.rm = TRUE) - 1
  x1_max <- max(train_data[[feature1]], na.rm = TRUE) + 1
  x2_min <- min(train_data[[feature2]], na.rm = TRUE) - 1
  x2_max <- max(train_data[[feature2]], na.rm = TRUE) + 1
  
  grid_x1 <- seq(x1_min, x1_max, by = h)
  grid_x2 <- seq(x2_min, x2_max, by = h)
  grid_df <- expand.grid(grid_x1, grid_x2)
  names(grid_df) <- c(feature1, feature2)
  
  for (col in setdiff(names(train_data), c(target_col, feature1, feature2))) {
    if (is.numeric(train_data[[col]]) || is.integer(train_data[[col]])) {
      grid_df[[col]] <- median(train_data[[col]], na.rm = TRUE)
    } else if (is.factor(train_data[[col]])) {
      mode_value <- names(sort(table(train_data[[col]]), decreasing = TRUE))[1]
      grid_df[[col]] <- factor(mode_value, levels = levels(train_data[[col]]))
    } else {
      grid_df[[col]] <- train_data[[col]][1]
    }
  }
  
  grid_df$pred <- predict(model, grid_df)
  grid_df$pred_num <- ifelse(grid_df$pred == "yes", 1, 0)
  
  ggplot() +
    geom_tile(data = grid_df, aes(x = !!sym(feature1), y = !!sym(feature2), fill = factor(pred_num)), alpha = 0.3) +
    geom_point(data = train_data, aes(x = !!sym(feature1), y = !!sym(feature2), color = !!sym(target_col)), size = 2) +
    labs(title = "KNN - Biên Quyết Định",
         x = feature1,
         y = feature2) +
    scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
    scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
    theme_minimal()
}

plot_confusion_matrix <- function(cm, model_name) {
  confusion_table <- as.data.frame(cm$table)
  
  ggplot(confusion_table, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), color = "black") +
    geom_text(aes(label = Freq), vjust = 1, size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = paste("Ma Trận Nhầm Lẫn -", model_name),
         x = "Giá Trị Thực",
         y = "Giá Trị Dự Báo") +
    theme_minimal()
}

plot_metrics_comparison <- function(comparison_df) {
  metrics_melted <- data.frame(
    Model = rep(rownames(comparison_df), ncol(comparison_df)),
    Metric = rep(colnames(comparison_df), each = nrow(comparison_df)),
    Value = c(as.matrix(comparison_df))
  )
  
  ggplot(metrics_melted, aes(x = Model, y = Value, fill = Metric)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5) +
    labs(title = "So Sánh Hiệu Suất Mô Hình",
         x = "Mô Hình",
         y = "Điểm Số") +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

print_model_summary <- function(model, model_name) {
  cat("\n", strrep("=", 70), "\n")
  cat("MÔ HÌNH:", model_name, "\n")
  cat(strrep("=", 70), "\n")
  cat("\nTham Số Tốt Nhất:\n")
  print(model$bestTune)
  cat("\nKết Quả CV:\n")
  print(model$results)
}
