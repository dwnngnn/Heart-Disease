library(ggplot2)
library(dplyr)
library(corrplot)


## 1. Tổng quan dữ liệu

df <- read.csv("heart_cleveland_raw.csv")
dim(df)
head(df)
str(df)


## 2. Thống kê mô tả

summary(df)


## 3. Kiểm tra chất lượng
# Missing values
colSums(is.na(df))


# Outlier (giá trị >= 999 hoặc <= -999)

sapply(df[sapply(df, is.numeric)], function(x) sum(abs(x) >= 999, na.rm = TRUE))

# Trùng lặp
sum(duplicated(df))


## 4. Phân phối biến số

par(mfrow = c(3,4))
for(col in names(df[sapply(df, is.numeric)])) {
  hist(df[[col]], main = col, xlab = col, col = "steelblue")
}


## 5. Quan hệ với biến mục tiêu

# Cân bằng target
table(df$target)
prop.table(table(df$target))

# Boxplot so sánh
par(mfrow = c(2,3))
for(col in c("age", "chol", "trestbps", "thalach", "oldpeak",  "ca")) {
  boxplot(df[[col]] ~ df$target, main = col, col = c("green", "red"))
}

## 6. Tương quan
cor_matrix <- cor(df[sapply(df, is.numeric)], use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")

cat("Số dòng:", nrow(df), "\n")
cat("Số biến:", ncol(df), "\n")
cat("Missing:", sum(is.na(df)), "\n")
cat("Tỷ lệ mắc bệnh:", mean(df$heart_disease)*100, "%\n")
