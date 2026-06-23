# Install packages ONCE if needed:
#install.packages(c("quantmod","xgboost", "DALEX", "iml", "data.table"))
# 1) Use the main CRAN cloud mirror
#options(repos = c(CRAN = "https://cloud.r-project.org"))

# 2) Try installing from source instead of binary
#install.packages("xgboost", type = "source")
library(quantmod)   # download financial data
library(dplyr)      # data manipulation
library(lubridate)  # date handling
library(ggplot2)    # plotting
library(xgboost)    # XGBoost model
library(DALEX)      # model explainers: permutation, PDP, ALE, SHAP
library(iml)        # alternative for PDP/ALE (we'll use DALEX mainly)
library(data.table) # for easier matrix / data conversion

# Define ticker and time period
ticker <- "^IXIC"  # NASDAQ Composite Index

end_date   <- Sys.Date()
start_date <- end_date - years(10)

# Download from Yahoo Finance
getSymbols(ticker, src = "yahoo", from = start_date, to = end_date, auto.assign = TRUE)

# The object name will be something like "^IXIC" with dots, so we use get()
nasdaq_raw <- IXIC

# Quick look
head(nasdaq_raw)

# Convert xts to data frame
nasdaq_df <- data.frame(
  date   = index(nasdaq_raw),
  coredata(nasdaq_raw)
)

# Quick NA check
colSums(is.na(nasdaq_df)) #no NA

# Clean column names
colnames(nasdaq_df) <- c("date", "open", "high", "low", "close", "volume", "adjusted")

# Arrange by date just in case
nasdaq_df <- nasdaq_df %>% arrange(date)
#---------------NEW ESTIMATORS-----------#
nasdaq_df <- nasdaq_df %>%
  arrange(date) %>%
  mutate(
    daily_return = adjusted / lag(adjusted) - 1,
    vol_30d = zoo::rollapply(daily_return, width = 30,
                             FUN = sd, align = "right", fill = NA),
    ma7  = zoo::rollmean(adjusted, 7,  fill = NA, align = "right"),
    ma30 = zoo::rollmean(adjusted, 30, fill = NA, align = "right"),
    volume_change = volume / lag(volume),
    abs_return    = abs(daily_return),
    
    # NEW FEATURES
    lag_vol_1  = dplyr::lag(vol_30d, 1),
    lag_vol_5  = dplyr::lag(vol_30d, 5),
    lag_vol_10 = dplyr::lag(vol_30d, 10),
    sq_return  = daily_return^2
  )

# Remove first rows with NA due to lag/rolling
nasdaq_df <- nasdaq_df %>%
  tidyr::drop_na()

# Quick check
summary(nasdaq_df)

# Keep only the columns we will use
data_model <- nasdaq_df %>%
  select(
    date,
    vol_30d,          # target
    daily_return,
    abs_return,
    ma7, ma30,
    volume, volume_change,
    lag_vol_1, lag_vol_5, lag_vol_10,
    sq_return
  )


# Define proportion for train/test
train_ratio <- 0.8
n <- nrow(data_model)
train_size <- floor(train_ratio * n)

train_data <- data_model[1:train_size, ]
test_data  <- data_model[(train_size + 1):n, ]

# X (features) and y (target)
X_train <- train_data %>% select(-date, -vol_30d)
y_train <- train_data$vol_30d

X_test  <- test_data %>% select(-date, -vol_30d)
y_test  <- test_data$vol_30d

# Convert to matrix (required by xgboost)
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

# Create DMatrix (xgboost internal data structure)
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train)
dtest  <- xgb.DMatrix(data = X_test_mat, label = y_test)

# Set a simple parameter grid
params <- list(
  objective = "reg:squarederror", # regression
  eta       = 0.05,               # learning rate
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train model
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data   = dtrain,
  nrounds = 300,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 50,
  early_stopping_rounds = 30
)

xgb_model

# Predictions
y_pred_test <- predict(xgb_model, newdata = dtest)

# RMSE and MAE
rmse <- sqrt(mean((y_test - y_pred_test)^2))
mae  <- mean(abs(y_test - y_pred_test))

rmse
mae

# Plot actual vs predicted volatility
plot_df <- data.frame(
  date      = test_data$date,
  actual    = y_test,
  predicted = y_pred_test
)

ggplot(plot_df, aes(x = date)) +
  geom_line(aes(y = actual), size = 0.7) +
  geom_line(aes(y = predicted), linetype = "dashed") +
  labs(
    title = "Actual vs Predicted 30-Day Volatility (Test Set)",
    y = "Volatility",
    x = "Date"
  ) +
  theme_minimal()

#--------------------------------------------
# Turn training data into data frame (without target)
X_train_df <- as.data.frame(X_train)
X_test_df  <- as.data.frame(X_test)

# Create explainer
explainer_xgb <- explain(
  model = xgb_model,
  data  = X_train_df,
  y     = y_train,
  label = "XGBoost Volatility Model",
  predict_function = function(m, newdata) {
    predict(m, newdata = as.matrix(newdata))
  }
)
# Permutation feature importance
fi <- model_parts(explainer_xgb, type = "variable_importance")

fi

plot(fi) +
  ggtitle("Permutation Feature Importance - XGBoost")
#-----------------PDP'S--------------------------#
# PDP for daily_return - optional 
pdp_return <- model_profile(explainer_xgb,
                            variables = "daily_return",
                            type = "partial")

plot(pdp_return) +
  ggtitle("PDP - Effect of Daily Return on Predicted Volatility")

# PDP for ma30
pdp_ma30 <- model_profile(explainer_xgb,
                          variables = "ma30",
                          type = "partial")

plot(pdp_ma30) +
  ggtitle("PDP - Effect of 30-Day Moving Average on Predicted Volatility")

# PDP for volume_change
pdp_volchange <- model_profile(explainer_xgb,
                               variables = "volume_change",
                               type = "partial")

plot(pdp_volchange) +
  ggtitle("PDP - Effect of Volume Change on Predicted Volatility")

# PDP - lagged volatility 1 day
pdp_lag1 <- model_profile(
  explainer_xgb,
  variables = "lag_vol_1",
  type = "partial"
)

plot(pdp_lag1) +
  ggtitle("PDP - Effect of 1-Day Lagged Volatility")

# PDP - lagged volatility 10 days
pdp_lag10 <- model_profile(
  explainer_xgb,
  variables = "lag_vol_10",
  type = "partial"
)

plot(pdp_lag10) +
  ggtitle("PDP - Effect of 10-Day Lagged Volatility")

# PDP - squared return
pdp_sqret <- model_profile(
  explainer_xgb,
  variables = "sq_return",
  type = "partial"
)

plot(pdp_sqret) +
  ggtitle("PDP - Effect of Squared Return on Predicted Volatility")

#-------------ALE's-----------#
# ALE for daily_return
ale_return <- model_profile(explainer_xgb,
                            variables = "daily_return",
                            type = "accumulated")

plot(ale_return) +
  ggtitle("ALE - Effect of Daily Return on Predicted Volatility")

# ALE for ma30
ale_ma30 <- model_profile(explainer_xgb,
                          variables = "ma30",
                          type = "accumulated")

plot(ale_ma30) +
  ggtitle("ALE - Effect of 30-Day MA on Predicted Volatility")

# ALE - 1-day lagged volatility
ale_lag1 <- model_profile(
  explainer_xgb,
  variables = "lag_vol_1",
  type = "accumulated"
)

plot(ale_lag1) +
  ggtitle("ALE - Effect of 1-Day Lagged Volatility")

# ALE - squared return
ale_sqret <- model_profile(
  explainer_xgb,
  variables = "sq_return",
  type = "accumulated"
)

plot(ale_sqret) +
  ggtitle("ALE - Effect of Squared Return")



# Take a sample from test data for SHAP (to save time)
set.seed(123)
shap_sample_idx <- sample(seq_len(nrow(X_test_df)), size = min(200, nrow(X_test_df)))
X_shap <- X_test_df[shap_sample_idx, , drop = FALSE]
y_shap <- y_test[shap_sample_idx]

# Compute SHAP values for the sample
# Compute SHAP values one observation at a time
shap_values_list <- lapply(1:nrow(X_shap), function(i) {
  predict_parts(
    explainer_xgb,
    new_observation = X_shap[i, , drop = FALSE],
    type = "shap"
  )
})


# shap_values_list is a list of explanations, one per observation.
# Convert to data frame to summarize:

shap_df <- do.call(rbind, lapply(seq_along(shap_values_list), function(i) {
  tmp <- as.data.frame(shap_values_list[[i]])  # <-- use res itself
  tmp$observation_id <- i                     # add id
  tmp
}))

# Filter out the intercept / baseline
library(dplyr)
library(stringr)

shap_global <- shap_df %>%
  # remove baseline rows if still present
  filter(variable != "_baseline_") %>%
  # extract feature name (before the " = ")
  mutate(feature = sub(" =.*", "", variable)) %>%
  group_by(feature) %>%
  summarize(mean_abs_shap = mean(abs(contribution))) %>%
  arrange(desc(mean_abs_shap))

shap_global


# Plot global SHAP importance
ggplot(shap_global, aes(x = reorder(feature, mean_abs_shap),
                        y = mean_abs_shap)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Global SHAP Importance (Mean |SHAP|)",
    x = "Feature",
    y = "Mean |SHAP|"
  ) +
  theme_minimal()

