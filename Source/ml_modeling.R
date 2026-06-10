# ============================================================================
# Main Modeling Script - Dự Báo Bệnh Tim
# ============================================================================

# Set working directory
setwd("C:\\TaiLieuHocTap\\R\\HD\\Dataset")

# Load module
source("ml_models.R")

# ============================================================================
# CÁC ĐẶC TRƯNG VÀ CHUẨN BỊ DỮ LIỆU
# ============================================================================
set.seed(42)
df <- read.csv("heart_disease.csv")
df$target <- factor(df$target, levels = c(0, 1), labels = c("no", "yes"))

# Split train/test (90/10)
idx <- createDataPartition(df$target, p = 0.9, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

predictors <- setdiff(names(df), "target")

cat("Thông Tin Dữ Liệu:\n")
cat("- Tổng mẫu:", nrow(df), "\n")
cat("- Mẫu huấn luyện:", nrow(train), "\n")
cat("- Mẫu kiểm thử:", nrow(test), "\n")
cat("- Đặc trưng:", length(predictors), "\n\n")

# 5.2. LOGISTIC REGRESSION
cat("Xây dựng Logistic Regression (5.2)...\n")
mod_lr <- build_logistic_regression(
  train_data = train,
  predictors = predictors,
  target_col = "target",
  cv_folds = 5,
  seed = 42
)
cat("✓ Logistic Regression xây dựng thành công\n\n")

# ============================================================================
# 5.3. RANDOM FOREST MODEL
cat("Building Random Forest Model (5.3)...\n")
mod_rf <- build_random_forest(
  train_data = train,
  predictors = predictors,
  target_col = "target",
  cv_folds = 5,
  seed = 42
)
cat("✓ Random Forest model built successfully\n\n")
# 5.5. KNN MODEL
# ============================================================================
cat("Building KNN Model (5.5)...\n")
mod_knn <- build_knn(
  train_data = train,
  predictors = predictors,
  target_col = "target",
  cv_folds = 5,
  seed = 42
)
cat("✓ KNN xây dựng thành công\n\n")

# ============================================================================
# ĐÁNH GIÁ MÔ HÌNH
# ============================================================================
cat("Đánh giá trên tập kiểm thử...\n\n")

models_list <- list(
  LogisticRegression = mod_lr,
  RandomForest = mod_rf,
  KNN = mod_knn
)

comparison <- compare_models(
  models_list = models_list,
  test_data = test[, predictors],
  test_target = test$target,
  model_names = names(models_list)
)

# Hiển thị chỉ số
cat("So Sánh Hiệu Suất Mô Hình:\n")
print(round(comparison$metrics, 4))

# Hiển thị ma trận nhầm lẫn
cat("\n", strrep("=", 80), "\n")
for (name in names(comparison$confusion_matrices)) {
  cat("\nMa Trận Nhầm Lẫn -", name, "\n")
  print(comparison$confusion_matrices[[name]]$table)
  cat("\nĐộ Nhạy:", comparison$confusion_matrices[[name]]$byClass["Sensitivity"], "\n")
  cat("Độ Đặc Hiệu:", comparison$confusion_matrices[[name]]$byClass["Specificity"], "\n")
}

# Find best model
f1_scores <- comparison$metrics["F1", ]
best_model_idx <- which.max(f1_scores)
best_model_name <- names(f1_scores)[best_model_idx]
best_model <- models_list[[best_model_name]]

cat("\n", strrep("=", 80), "\n")
cat("VISUALIZATIONS\n")
cat(strrep("=", 80), "\n\n")

p1 <- plot_logistic_regression(mod_lr, test[, predictors], test$target)
print(p1)

p2 <- plot_random_forest(mod_rf)
print(p2)

p3 <- plot_confusion_matrix(comparison$confusion_matrices$LogisticRegression, "Logistic Regression")
print(p3)

p4 <- plot_confusion_matrix(comparison$confusion_matrices$RandomForest, "Random Forest")
print(p4)

p5 <- plot_confusion_matrix(comparison$confusion_matrices$KNN, "KNN")
print(p5)

p6 <- plot_metrics_comparison(comparison$metrics)
print(p6)

# KNN - Biên quyết định với 2 đặc trưng chính
p7 <- plot_knn_boundary(mod_knn, train, "target", predictors[1], predictors[2])
print(p7)

cat("\n", strrep("=", 80), "\n")
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  cat("Dự báo trên dữ liệu mới...\n")
  new_data <- read.csv(args[1])
  if ("target" %in% names(new_data)) {
    new_data$target <- NULL
  }
  
  p_new <- predict(best_model, new_data[, predictors])
  write.csv(
    data.frame(prediction = p_new),
    "predictions.csv",
    row.names = FALSE
  )
  cat("✓ Kết quả lưu vào predictions.csv\n")
}