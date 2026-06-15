
# FINAL PROJECT – FORECASTING AND SIMULATIONS
# Topic: Conditional Forecast of Rain Tomorrow
# Model: Logit + Conditional Ex-Ante Forecast


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
  "knitr",
  "kableExtra"
)

install.packages(setdiff(packages, rownames(installed.packages())))
lapply(packages, library, character.only = TRUE)

# ==========================================================
# 2. LOAD DATA
# ==========================================================

train <- read.csv("Weather-Training-Data.csv")
test  <- read.csv("Weather-Test-Data.csv")

selected_cities <- c("Sydney", "Perth", "Darwin", "Hobart", "AliceSprings")

selected_cities %in% unique(train$Location)
selected_cities %in% unique(test$Location)

train_cities <- subset(train, Location %in% selected_cities)
test_cities  <- subset(test, Location %in% selected_cities)

# ==========================================================
# 3. DATA CLEANING
# ==========================================================

train_cities <- train_cities %>%
  mutate(
    RainToday = ifelse(tolower(trimws(RainToday)) == "yes", 1, 0),
    RainTomorrow = as.numeric(RainTomorrow)
  )

test_cities <- test_cities %>%
  mutate(
    RainToday = ifelse(tolower(trimws(RainToday)) == "yes", 1, 0)
  )

model_vars <- c(
  "Location",
  "RainTomorrow",
  "Rainfall",
  "Sunshine",
  "Humidity3pm",
  "Pressure3pm",
  "Cloud3pm",
  "WindGustSpeed",
  "Temp3pm",
  "RainToday"
)

test_vars <- c(
  "Location",
  "Rainfall",
  "Sunshine",
  "Humidity3pm",
  "Pressure3pm",
  "Cloud3pm",
  "WindGustSpeed",
  "Temp3pm",
  "RainToday"
)

train_clean <- train_cities %>%
  select(all_of(model_vars)) %>%
  na.omit()

test_clean <- test_cities %>%
  select(all_of(test_vars)) %>%
  na.omit()

colSums(is.na(train_clean))
colSums(is.na(test_clean))

# ==========================================================
# 4. DESCRIPTIVE STATISTICS
# ==========================================================

summary(train_clean)

table(train_clean$RainTomorrow)
prop.table(table(train_clean$RainTomorrow))

desc_stats <- train_clean %>%
  summarise(across(where(is.numeric), list(
    mean = mean,
    sd = sd,
    min = min,
    max = max
  )))

kable(desc_stats, caption = "Descriptive Statistics")

# ==========================================================
# 5. EXPLORATORY VISUALISATIONS
# ==========================================================

p1 <- ggplot(train_clean, aes(x = factor(RainTomorrow))) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Distribution of Rain Tomorrow",
    x = "Rain Tomorrow",
    y = "Count"
  ) +
  theme_minimal()

print(p1)

p2 <- ggplot(train_clean, aes(x = Humidity3pm, y = RainTomorrow)) +
  geom_jitter(height = 0.05, alpha = 0.1, color = "darkblue") +
  geom_smooth(
    method = "glm",
    method.args = list(family = "binomial"),
    se = TRUE,
    color = "red"
  ) +
  labs(
    title = "Humidity at 3pm and Probability of Rain Tomorrow",
    x = "Humidity at 3pm",
    y = "Rain Tomorrow"
  ) +
  theme_minimal()

print(p2)

cor_matrix <- train_clean %>%
  select(where(is.numeric)) %>%
  cor()

print(round(cor_matrix, 3))

# ==========================================================
# 6. LOGIT MODEL
# ==========================================================

logit_model <- glm(
  RainTomorrow ~ Rainfall + Sunshine + Humidity3pm + Pressure3pm +
    Cloud3pm + WindGustSpeed + Temp3pm + RainToday + factor(Location),
  data = train_clean,
  family = binomial(link = "logit")
)

summary(logit_model)

coef_table <- tidy(logit_model)

kable(
  coef_table,
  caption = "Logit Model Coefficients",
  digits = 4
)

# ==========================================================
# 7. ODDS RATIOS
# ==========================================================

odds_ratios <- exp(coef(logit_model))
conf_int <- exp(confint(logit_model))
p_values <- summary(logit_model)$coefficients[, 4]

or_table <- data.frame(
  Variable = names(odds_ratios),
  Odds_Ratio = round(odds_ratios, 3),
  CI_Lower = round(conf_int[, 1], 3),
  CI_Upper = round(conf_int[, 2], 3),
  P_Value = round(p_values, 4)
)

or_table_report <- or_table %>%
  filter(!grepl("factor\\(Location\\)", Variable))

or_table_report

or_table_report %>%
  kable(
    caption = "Odds Ratios from the Logit Model",
    digits = 3
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover"),
    full_width = FALSE
  )

# Odds ratio plot without location dummies

logit_model_num <- glm(
  RainTomorrow ~ Rainfall + Sunshine + Humidity3pm + Pressure3pm +
    Cloud3pm + WindGustSpeed + Temp3pm + RainToday,
  data = train_clean,
  family = binomial(link = "logit")
)

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
    subtitle = "Odds Ratios from Logistic Regression",
    x = "Weather Variable",
    y = "Odds Ratio"
  ) +
  theme_minimal()

print(p_odds)

# ==========================================================
# 8. MODEL VERIFICATION
# ==========================================================

model_aic <- AIC(logit_model)
model_bic <- BIC(logit_model)

model_aic
model_bic

pseudo_r2 <- pR2(logit_model)
pseudo_r2

# VIF cannot be calculated properly with many location dummies,
# so VIF is calculated for the numerical version of the model.

vif_values <- vif(logit_model_num)
print(vif_values)

# ==========================================================
# 9. CLASSIFICATION VERIFICATION
# ==========================================================

train_clean$pred_prob <- predict(logit_model, type = "response")
train_clean$pred_class <- ifelse(train_clean$pred_prob >= 0.5, 1, 0)

conf_matrix <- confusionMatrix(
  factor(train_clean$pred_class, levels = c("0", "1")),
  factor(train_clean$RainTomorrow, levels = c("0", "1")),
  positive = "1"
)

conf_matrix

roc_train <- roc(train_clean$RainTomorrow, train_clean$pred_prob)

plot(
  roc_train,
  main = "ROC Curve – Training Data",
  col = "blue",
  lwd = 2
)

auc_train <- auc(roc_train)
auc_train

brier_train <- mean((train_clean$pred_prob - train_clean$RainTomorrow)^2)
brier_train

# ==========================================================
# 10. STATIONARITY TESTS
# ==========================================================

train_sydney_tests <- train_clean %>%
  filter(Location == "Sydney")

adf.test(train_sydney_tests$Humidity3pm)
kpss.test(train_sydney_tests$Humidity3pm)

ggtsdisplay(
  train_sydney_tests$Humidity3pm,
  main = "Time Series, ACF and PACF for Humidity3pm – Sydney"
)

# ==========================================================
# 11. CONDITIONAL EX-ANTE FORECAST USING TEST DATA
# ==========================================================

test_clean$Predicted_Probability <- predict(
  logit_model,
  newdata = test_clean,
  type = "response"
)

test_clean$Predicted_Class <- ifelse(
  test_clean$Predicted_Probability >= 0.5,
  1,
  0
)

forecast_7days <- test_clean %>%
  group_by(Location) %>%
  slice(1:7) %>%
  mutate(Day = paste0("Day ", row_number())) %>%
  ungroup()

forecast_table <- forecast_7days %>%
  mutate(
    Predicted_Probability = round(Predicted_Probability * 100, 2)
  ) %>%
  select(Day, Location, Predicted_Probability) %>%
  pivot_wider(
    names_from = Location,
    values_from = Predicted_Probability
  )

forecast_table

forecast_table %>%
  kable(
    caption = "7-Day Conditional Forecast of Rain Probability (%)",
    digits = 2
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover"),
    full_width = FALSE
  )

p3 <- ggplot(
  forecast_7days,
  aes(
    x = Day,
    y = Predicted_Probability,
    group = Location,
    color = Location
  )
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  labs(
    title = "Conditional Ex-Ante Forecast Across Selected Australian Cities",
    subtitle = "Probability of Rain Tomorrow, 7-Day Horizon",
    x = "Forecast Horizon",
    y = "Probability of Rain"
  ) +
  theme_minimal()

print(p3)

# ==========================================================
# 12. OPTIONAL ARIMA FORECAST OF EXPLANATORY VARIABLES
# ==========================================================

forecast_x <- function(series, h) {
  ts_data <- ts(series, frequency = 1)
  model <- auto.arima(ts_data)
  fc <- forecast(model, h = h)
  
  print(Box.test(model$residuals, type = "Ljung-Box"))
  
  return(list(
    mean = as.numeric(fc$mean),
    model = model
  ))
}

h_forecast <- 7

future_list <- list()

for (city in selected_cities) {
  
  city_data <- train_clean %>%
    filter(Location == city)
  
  fc_rainfall <- forecast_x(city_data$Rainfall, h_forecast)
  fc_sunshine <- forecast_x(city_data$Sunshine, h_forecast)
  fc_humidity <- forecast_x(city_data$Humidity3pm, h_forecast)
  fc_pressure <- forecast_x(city_data$Pressure3pm, h_forecast)
  fc_cloud <- forecast_x(city_data$Cloud3pm, h_forecast)
  fc_wind <- forecast_x(city_data$WindGustSpeed, h_forecast)
  fc_temp <- forecast_x(city_data$Temp3pm, h_forecast)
  
  future_city <- data.frame(
    Location = city,
    Day = paste0("Day ", 1:h_forecast),
    RainToday = rep(tail(city_data$RainToday, 1), h_forecast),
    Rainfall = fc_rainfall$mean,
    Sunshine = fc_sunshine$mean,
    Humidity3pm = fc_humidity$mean,
    Pressure3pm = fc_pressure$mean,
    Cloud3pm = fc_cloud$mean,
    WindGustSpeed = fc_wind$mean,
    Temp3pm = fc_temp$mean
  )
  
  future_list[[city]] <- future_city
}

future_x <- bind_rows(future_list)

future_x$Forecast_Probability <- predict(
  logit_model,
  newdata = future_x,
  type = "response"
)

future_x$Forecast_Class <- ifelse(
  future_x$Forecast_Probability >= 0.5,
  1,
  0
)

future_x$Forecast_Label <- ifelse(
  future_x$Forecast_Class == 1,
  "Rain expected",
  "No rain expected"
)

final_forecast <- future_x %>%
  select(
    Location,
    Day,
    Forecast_Probability,
    Forecast_Class,
    Forecast_Label,
    everything()
  )

print(final_forecast[, 1:5])

write.csv(
  final_forecast,
  "Final_Ex_Ante_Rain_Forecast.csv",
  row.names = FALSE
)

p4 <- ggplot(
  final_forecast,
  aes(
    x = Day,
    y = Forecast_Probability,
    group = Location,
    color = Location
  )
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  labs(
    title = "ARIMA-Based Conditional Ex-Ante Forecast",
    subtitle = "Forecast Probability of Rain Tomorrow, 7-Day Horizon",
    x = "Forecast Horizon",
    y = "Forecast Probability"
  ) +
  theme_minimal()

print(p4)

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
    "Number of Test Forecast Observations"
  ),
  Value = c(
    round(model_aic, 3),
    round(model_bic, 3),
    round(pseudo_r2["McFadden"], 4),
    round(as.numeric(auc_train), 4),
    round(brier_train, 4),
    nrow(train_clean),
    nrow(test_clean)
  )
)

summary_table %>%
  kable(
    caption = "Summary of Model Verification Results",
    digits = 4
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover"),
    full_width = FALSE
  )

