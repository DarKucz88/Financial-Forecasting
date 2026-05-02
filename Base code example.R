############################################################
# FINAL PROJECT – FORECASTING AND SIMULATIONS
# Topic: Conditional Forecast of Rain Tomorrow
# Model: Logit + Explanatory Variable Forecasts
############################################################

# ==========================================================
# 1. PACKAGES
# ==========================================================

packages <- c(
  "tidyverse",
  "forecast",
  "tseries",
  "lmtest",
  "car",
  "pscl",
  "pROC",
  "caret",
  "broom",
  "knitr"
)

install.packages(setdiff(packages, rownames(installed.packages())))
lapply(packages, library, character.only = TRUE)

# ==========================================================
# 2. LOAD DATA
# ==========================================================

train <- read.csv("Weather-Training-Data.csv")
test  <- read.csv("Weather-Test-Data.csv")

str(train)
str(test)

# Training data has RainTomorrow
# Test data does NOT have RainTomorrow, so test data is used for ex-ante forecast only

# ==========================================================
# 3. DATA CLEANING
# ==========================================================

train <- train %>%
  mutate(
    RainToday = ifelse(RainToday == "Yes", 1, 0),
    RainTomorrow = ifelse(RainTomorrow == "Yes", 1, 0)
  )

test <- test %>%
  mutate(
    RainToday = ifelse(RainToday == "Yes", 1, 0)
  )

# Variables used in the model
model_vars <- c(
  "RainTomorrow",
  "RainToday",
  "Rainfall",
  "Sunshine",
  "Humidity3pm",
  "Pressure3pm",
  "Cloud3pm",
  "WindGustSpeed",
  "Temp3pm"
)

test_vars <- c(
  "RainToday",
  "Rainfall",
  "Sunshine",
  "Humidity3pm",
  "Pressure3pm",
  "Cloud3pm",
  "WindGustSpeed",
  "Temp3pm"
)

train_clean <- train %>%
  select(all_of(model_vars)) %>%
  na.omit()

test_clean <- test %>%
  select(all_of(test_vars)) %>%
  na.omit()

# Check missing values
colSums(is.na(train_clean))
colSums(is.na(test_clean))

# ==========================================================
# 4. DESCRIPTIVE STATISTICS
# ==========================================================

summary(train_clean)

# Distribution of dependent variable
table(train_clean$RainTomorrow)
prop.table(table(train_clean$RainTomorrow))

# Descriptive table
desc_stats <- train_clean %>%
  summarise(across(everything(), list(
    mean = mean,
    sd = sd,
    min = min,
    max = max
  )))

kable(desc_stats)

# ==========================================================
# 5. EXPLORATORY VISUALISATIONS
# ==========================================================

ggplot(train_clean, aes(x = factor(RainTomorrow))) +
  geom_bar() +
  labs(
    title = "Distribution of Rain Tomorrow",
    x = "Rain Tomorrow",
    y = "Number of Observations"
  )

ggplot(train_clean, aes(x = Humidity3pm, y = RainTomorrow)) +
  geom_jitter(height = 0.05, alpha = 0.3) +
  geom_smooth(
    method = "glm",
    method.args = list(family = "binomial"),
    se = TRUE
  ) +
  labs(
    title = "Humidity at 3pm and Probability of Rain Tomorrow",
    x = "Humidity at 3pm",
    y = "Rain Tomorrow"
  )

ggplot(train_clean, aes(x = Pressure3pm, y = RainTomorrow)) +
  geom_jitter(height = 0.05, alpha = 0.3) +
  geom_smooth(
    method = "glm",
    method.args = list(family = "binomial"),
    se = TRUE
  ) +
  labs(
    title = "Pressure at 3pm and Probability of Rain Tomorrow",
    x = "Pressure at 3pm",
    y = "Rain Tomorrow"
  )

# Correlation matrix
cor_matrix <- cor(train_clean)
round(cor_matrix, 3)

# ==========================================================
# 6. ECONOMETRIC CAUSE-EFFECT MODEL
# ==========================================================

logit_model <- glm(
  RainTomorrow ~ RainToday + Rainfall + Sunshine +
    Humidity3pm + Pressure3pm + Cloud3pm +
    WindGustSpeed + Temp3pm,
  data = train_clean,
  family = binomial(link = "logit")
)

summary(logit_model)

# Coefficients table
coef_table <- tidy(logit_model)
kable(coef_table)

# Odds ratios
odds_ratios <- exp(coef(logit_model))
kable(data.frame(Variable = names(odds_ratios), Odds_Ratio = odds_ratios))

# ==========================================================
# 7. NUMERICAL MODEL VERIFICATION
# ==========================================================

# AIC and BIC
model_aic <- AIC(logit_model)
model_bic <- BIC(logit_model)

model_aic
model_bic

# Pseudo R-squared
pseudo_r2 <- pR2(logit_model)
pseudo_r2

# Multicollinearity
vif_values <- vif(logit_model)
vif_values

# ==========================================================
# 8. STOCHASTIC / CLASSIFICATION VERIFICATION
# ==========================================================

train_clean$pred_prob <- predict(logit_model, type = "response")
train_clean$pred_class <- ifelse(train_clean$pred_prob >= 0.5, 1, 0)

conf_matrix <- confusionMatrix(
  factor(train_clean$pred_class),
  factor(train_clean$RainTomorrow),
  positive = "1"
)

conf_matrix

# ROC curve and AUC
roc_train <- roc(train_clean$RainTomorrow, train_clean$pred_prob)

plot(
  roc_train,
  main = "ROC Curve – Training Data"
)

auc_train <- auc(roc_train)
auc_train

# Brier score
brier_train <- mean((train_clean$pred_prob - train_clean$RainTomorrow)^2)
brier_train

# ==========================================================
# 9. FORECAST EXPLANATORY VARIABLES
# ==========================================================

forecast_x <- function(series, h) {
  ts_data <- ts(series, frequency = 1)
  model <- auto.arima(ts_data)
  forecast_values <- forecast(model, h = h)
  
  return(list(
    model = model,
    forecast = forecast_values
  ))
}

h <- nrow(test_clean)

fc_rainfall <- forecast_x(train_clean$Rainfall, h)
fc_sunshine <- forecast_x(train_clean$Sunshine, h)
fc_humidity <- forecast_x(train_clean$Humidity3pm, h)
fc_pressure <- forecast_x(train_clean$Pressure3pm, h)
fc_cloud <- forecast_x(train_clean$Cloud3pm, h)
fc_wind <- forecast_x(train_clean$WindGustSpeed, h)
fc_temp <- forecast_x(train_clean$Temp3pm, h)

# Show selected ARIMA models
fc_rainfall$model
fc_sunshine$model
fc_humidity$model
fc_pressure$model
fc_cloud$model
fc_wind$model
fc_temp$model

# Plot selected forecasts
autoplot(fc_humidity$forecast) +
  labs(title = "Forecast of Humidity at 3pm")

autoplot(fc_pressure$forecast) +
  labs(title = "Forecast of Pressure at 3pm")

autoplot(fc_temp$forecast) +
  labs(title = "Forecast of Temperature at 3pm")

autoplot(fc_wind$forecast) +
  labs(title = "Forecast of Wind Gust Speed")

# ==========================================================
# 10. CONDITIONAL EX-ANTE FORECAST
# ==========================================================

# We combine:
# - observed RainToday from test data
# - forecasted explanatory variables from training data

future_x <- data.frame(
  RainToday = test_clean$RainToday,
  Rainfall = as.numeric(fc_rainfall$forecast$mean),
  Sunshine = as.numeric(fc_sunshine$forecast$mean),
  Humidity3pm = as.numeric(fc_humidity$forecast$mean),
  Pressure3pm = as.numeric(fc_pressure$forecast$mean),
  Cloud3pm = as.numeric(fc_cloud$forecast$mean),
  WindGustSpeed = as.numeric(fc_wind$forecast$mean),
  Temp3pm = as.numeric(fc_temp$forecast$mean)
)

# Predict probability of rain tomorrow
future_x$Forecast_Probability_RainTomorrow <- predict(
  logit_model,
  newdata = future_x,
  type = "response"
)

# Classification threshold
future_x$Forecast_Class_RainTomorrow <- ifelse(
  future_x$Forecast_Probability_RainTomorrow >= 0.5,
  1,
  0
)

head(future_x)

# ==========================================================
# 11. FINAL FORECAST OUTPUT
# ==========================================================

final_forecast <- future_x %>%
  mutate(
    Forecast_Label = ifelse(
      Forecast_Class_RainTomorrow == 1,
      "Rain expected",
      "No rain expected"
    )
  )

head(final_forecast, 20)

write.csv(
  final_forecast,
  "Final_Ex_Ante_Rain_Forecast.csv",
  row.names = FALSE
)

# ==========================================================
# 12. VISUALISE FINAL FORECAST
# ==========================================================

final_forecast$row_id <- 1:nrow(final_forecast)

ggplot(final_forecast, aes(x = row_id, y = Forecast_Probability_RainTomorrow)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  labs(
    title = "Ex-Ante Conditional Forecast of Rain Tomorrow",
    subtitle = "Forecast probability from estimated logit model",
    x = "Test Observation",
    y = "Forecast Probability"
  )

ggplot(final_forecast, aes(x = Forecast_Probability_RainTomorrow)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Distribution of Forecasted Rain Probabilities",
    x = "Forecast Probability",
    y = "Frequency"
  )

# ==========================================================
# 13. SUMMARY TABLE FOR REPORT
# ==========================================================

summary_table <- data.frame(
  Metric = c(
    "AIC",
    "BIC",
    "Pseudo R-squared: McFadden",
    "Training AUC",
    "Training Brier Score",
    "Number of Training Observations",
    "Number of Forecast Observations"
  ),
  Value = c(
    model_aic,
    model_bic,
    pseudo_r2["McFadden"],
    as.numeric(auc_train),
    brier_train,
    nrow(train_clean),
    nrow(test_clean)
  )
)

kable(summary_table)

# ==========================================================
# 14. TEXT FOR REPORT
# ==========================================================

cat("
The dependent variable in the model is RainTomorrow, which is binary and equals 1 if rain occurs tomorrow and 0 otherwise.

The model is estimated using the training dataset because this file contains both the explanatory variables and the dependent variable RainTomorrow.

The test dataset does not contain RainTomorrow. Therefore, it represents a future or unknown period and is used only for generating ex-ante conditional forecasts.

The logit model estimates the probability of rain tomorrow as a function of current weather conditions such as rainfall, sunshine, humidity, atmospheric pressure, cloud cover, wind gust speed, temperature, and whether it rained today.

Forecasts of the explanatory variables are generated using ARIMA models. These forecasted explanatory variables are then inserted into the estimated logit model to obtain conditional forecasts of the probability of rain tomorrow.

Because the test dataset does not include the actual value of RainTomorrow, the final test-period forecasts cannot be evaluated ex-post. Instead, model performance is verified on the training data using AIC, BIC, pseudo R-squared, ROC/AUC, confusion matrix, and Brier score.
")