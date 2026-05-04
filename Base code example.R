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
#We analyze all of Australia

train <- train %>%
  mutate(
    RainToday = ifelse(tolower(trimws(RainToday)) == "yes", 1, 0)
  )

test <- test %>%
  mutate(
    RainToday = ifelse(tolower(trimws(RainToday)) == "yes", 1, 0)
  )

model_vars <- c(
  "RainTomorrow", "Location", "RainToday", "Rainfall",
  "Sunshine", "Humidity3pm", "Pressure3pm",
  "Cloud3pm", "WindGustSpeed", "Temp3pm"
)

test_vars <- c(
  "Location", "RainToday", "Rainfall", "Sunshine",
  "Humidity3pm", "Pressure3pm", "Cloud3pm",
  "WindGustSpeed", "Temp3pm"
)

# Clean NA
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
  summarise(across(where(is.numeric), list(
    mean = mean,
    sd = sd,
    min = min,
    max = max
  )))

kable(desc_stats)

# ==========================================================
# 5. EXPLORATORY VISUALISATIONS
# ==========================================================

# We save the chart to the object to make sure it is displayed
p1 <- ggplot(train_clean, aes(x = factor(RainTomorrow))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Rain Tomorrow", x = "Rain Tomorrow", y = "Count")
print(p1)

p2 <- ggplot(train_clean, aes(x = Humidity3pm, y = RainTomorrow)) +
  geom_jitter(height = 0.05, alpha = 0.1, color="darkblue") +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = TRUE, color="red") +
  labs(title = "Humidity at 3pm and Probability of Rain Tomorrow", x = "Humidity at 3pm", y = "Rain Tomorrow")
print(p2)

# We select only numeric columns to skip the text "Location"
cor_matrix <- train_clean %>%
  select(where(is.numeric)) %>%
  cor()
print(round(cor_matrix, 3))

# ==========================================================
# 6. ECONOMETRIC CAUSE-EFFECT MODEL
# ==========================================================

logit_model <- glm(
  RainTomorrow ~ Location + RainToday + Rainfall + Sunshine +
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
or_table <- data.frame(Variable = names(odds_ratios), Odds_Ratio = odds_ratios)

logit_model_num <- glm(
  RainTomorrow ~ RainToday + Rainfall + Sunshine + Humidity3pm + Pressure3pm + Cloud3pm + WindGustSpeed + Temp3pm,
  data = train_clean, family = binomial(link = "logit")
)

# We extract numerical coefficients, without localization (for the readability of the graph)
or_data <- tidy(logit_model_num, exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    Impact = ifelse(estimate > 1, "Increases Rain Risk", "Decreases Rain Risk")
  )

p_odds <- ggplot(or_data, aes(x = reorder(term, estimate), y = estimate, color = Impact)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2, linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  coord_flip() +
  labs(
    title = "Impact of Weather Variables on Rain Probability",
    subtitle = "Odds Ratios from Logistic Regression (Values > 1 increase risk)",
    x = "Weather Variable",
    y = "Odds Ratio"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("Increases Rain Risk" = "firebrick", "Decreases Rain Risk" = "forestgreen"))

print(p_odds)

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
vif_values <- vif(logit_model_num)
print(vif_values)

# ==========================================================
# 8. STOCHASTIC / CLASSIFICATION VERIFICATION
# ==========================================================

train_clean$pred_prob <- predict(logit_model, type = "response")
train_clean$pred_class <- ifelse(train_clean$pred_prob >= 0.5, 1, 0)

conf_matrix <- confusionMatrix(
  factor(train_clean$pred_class, levels = c("0", "1")),
  factor(train_clean$RainTomorrow, levels = c("0", "1")),
  positive = "1"
)

conf_matrix

# ROC curve and AUC
roc_train <- roc(train_clean$RainTomorrow, train_clean$pred_prob)
plot(
  roc_train,
  main = "ROC Curve – Training Data", col="blue", lwd=2
)

auc_train <- auc(roc_train)
auc_train

# Brier score
brier_train <- mean((train_clean$pred_prob - train_clean$RainTomorrow)^2)
brier_train

# ==========================================================
# 8b. STATIONARITY TESTS 
# ==========================================================
# Verification on continuous time series (one city)
train_sydney_tests <- train_clean %>% filter(Location == "Sydney")

#Test ADF (H0: The series is non-stationary)
print(adf.test(train_sydney_tests$Humidity3pm))

# Test KPSS (H0: The series is stationary) 
print(kpss.test(train_sydney_tests$Humidity3pm))

p_acf <- ggtsdisplay(
  train_sydney_tests$Humidity3pm, 
  main = "Time Series, ACF and PACF for Humidity3pm (Sydney)",
  theme = theme_minimal()
)
print(p_acf)

# ==========================================================
# 9. FORECAST EXPLANATORY VARIABLES
# ==========================================================

forecast_x <- function(series, h, var_name) {
  ts_data <- ts(series, frequency = 1)
  model <- auto.arima(ts_data)
  forecast_values <- forecast(model, h = h)

  
  lb_test <- Box.test(model$residuals, type = "Ljung-Box")
  print(lb_test)
  
  return(as.numeric(forecast_values$mean))
}

h_forecast <- 7 # Generujemy prognozę (scenariusz) na następne 7 dni

#SCENARIO A: SYDNEY (Humid Climate)
train_syd <- train_clean %>% filter(Location == "Sydney")

fc_syd <- data.frame(
  Location = "Sydney",
  RainToday = rep(tail(train_syd$RainToday, 1), h_forecast),
  Rainfall = forecast_x(train_syd$Rainfall, h_forecast, "Syd_Rainfall"),
  Sunshine = forecast_x(train_syd$Sunshine, h_forecast, "Syd_Sunshine"),
  Humidity3pm = forecast_x(train_syd$Humidity3pm, h_forecast, "Syd_Hum"),
  Pressure3pm = forecast_x(train_syd$Pressure3pm, h_forecast, "Syd_Pres"),
  Cloud3pm = forecast_x(train_syd$Cloud3pm, h_forecast, "Syd_Cloud"),
  WindGustSpeed = forecast_x(train_syd$WindGustSpeed, h_forecast, "Syd_Wind"),
  Temp3pm = forecast_x(train_syd$Temp3pm, h_forecast, "Syd_Temp")
)

# SCENARIO B: ALICE SPRINGS (Arid/Desert Climate)
train_ali <- train_clean %>% filter(Location == "AliceSprings")

fc_ali <- data.frame(
  Location = "AliceSprings",
  RainToday = rep(tail(train_ali$RainToday, 1), h_forecast),
  Rainfall = forecast_x(train_ali$Rainfall, h_forecast, "Ali_Rainfall"),
  Sunshine = forecast_x(train_ali$Sunshine, h_forecast, "Ali_Sunshine"),
  Humidity3pm = forecast_x(train_ali$Humidity3pm, h_forecast, "Ali_Hum"),
  Pressure3pm = forecast_x(train_ali$Pressure3pm, h_forecast, "Ali_Pres"),
  Cloud3pm = forecast_x(train_ali$Cloud3pm, h_forecast, "Ali_Cloud"),
  WindGustSpeed = forecast_x(train_ali$WindGustSpeed, h_forecast, "Ali_Wind"),
  Temp3pm = forecast_x(train_ali$Temp3pm, h_forecast, "Ali_Temp")
)

#We combine weather data generated from ARIMA into one "future" database
future_x <- rbind(fc_syd, fc_ali)

# ==========================================================
# 10. CONDITIONAL EX-ANTE FORECAST
# ==========================================================

# We combine:
# - observed RainToday from test data
# - forecasted explanatory variables from training data


future_x$Forecast_Probability <- predict(
  logit_model,
  newdata = future_x,
  type = "response"
)

future_x$Forecast_Class <- ifelse(
  future_x$Forecast_Probability >= 0.5, 1, 0
)
# ==========================================================
# 11. FINAL FORECAST OUTPUT
# ==========================================================

final_forecast <- future_x %>%
  mutate(
    Forecast_Label = ifelse(
      Forecast_Class_RainTomorrow == 1,
      "Rain expected",
      "No rain expected"
    ), Day = rep(paste("Day", 1:h_forecast), 2)
  ) %>%
  select(Location, Day, Forecast_Probability, Forecast_Label, everything())
print(final_forecast[, 1:4])
head(final_forecast, 20)

write.csv(
  final_forecast,
  "Final_Ex_Ante_Rain_Forecast.csv",
  row.names = FALSE
)

#Chart comparing predictions for Sydney and Alice Springs
p3 <- ggplot(final_forecast, aes(x = Day, y = Forecast_Probability, group = Location, color = Location)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color="red") +
  labs(
    title = "Conditional Ex-Ante Forecast: Sydney vs Alice Springs",
    subtitle = "Probability of Rain Tomorrow (7-Day Horizon)",
    x = "Forecast Horizon",
    y = "Probability of Rain"
  ) +
  theme_minimal()

print(p3)
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
# Stochastic verification
# ==========================================================

#TESTS FOR STATIONARITY OF VARIABLES

# We perform tests for a sample variable (e.g. Humidity)
print(tseries::adf.test(train_clean$Humidity3pm))

#KPSS Test for Humidity3pm (H0: Stationarity)
print(tseries::kpss.test(train_clean$Humidity3pm))

#DIAGNOSTICS OF THE REST OF ARIMA MODELS

# We collect the generated models into one list
generated_models <- list(
  "Rainfall"      = fc_rainfall$model,
  "Sunshine"      = fc_sunshine$model,
  "Humidity3pm"   = fc_humidity$model,
  "Pressure3pm"   = fc_pressure$model,
  "Cloud3pm"      = fc_cloud$model,
  "WindGustSpeed" = fc_wind$model,
  "Temp3pm"       = fc_temp$model
)

#Ljung-Box test for model residuals:
# H0: No autocorrelation (residuals are white noise)
  print(Box.test(generated_models[[nazwa]]$residuals, type = "Ljung-Box"))
}


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
