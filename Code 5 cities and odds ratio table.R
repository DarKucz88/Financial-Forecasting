selected_cities <- c("Sydney", "Perth", "Darwin", "Hobart", "AliceSprings")
train <- read.csv("Weather-Training-Data.csv")
test <- read.csv("Weather-Test-Data.csv")
unique(train$Location)

selected_cities %in% unique(train$Location)
selected_cities %in% unique(test$Location)
train_cities <- subset(train, Location %in% selected_cities)
test_cities <- subset(test, Location %in% selected_cities)
train_cities$RainToday <- ifelse(train_cities$RainToday == "Yes", 1, 0)
train_cities$RainTomorrow <- as.numeric(train_cities$RainTomorrow)

test_cities$RainToday <- ifelse(test_cities$RainToday == "Yes", 1, 0)
vars <- c(
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

train_model <- train_cities[, vars]
train_model <- na.omit(train_model)

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

test_model <- test_cities[, test_vars]

logit_model <- glm(
  RainTomorrow ~ Rainfall + Sunshine + Humidity3pm + Pressure3pm +
    Cloud3pm + WindGustSpeed + Temp3pm + RainToday + factor(Location),
  data = train_model,
  family = binomial(link = "logit")
)

summary(logit_model)

test_model_clean <- na.omit(test_model)

test_model_clean$Predicted_Probability <- predict(
  logit_model,
  newdata = test_model_clean,
  type = "response"
)

library(dplyr)

forecast_7days <- test_model_clean %>%
  group_by(Location) %>%
  slice(1:7) %>%
  mutate(Day = paste0("Day ", row_number())) %>%
  ungroup()

forecast_table <- forecast_7days %>%
  select(Day, Location, Predicted_Probability) %>%
  tidyr::pivot_wider(
    names_from = Location,
    values_from = Predicted_Probability
  )

forecast_table


library(ggplot2)

ggplot(forecast_7days, aes(x = Day, y = Predicted_Probability,
                           group = Location, color = Location)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Conditional Ex-Ante Forecast Across Australia",
    subtitle = "Probability of Rain Tomorrow, 7-Day Horizon",
    x = "Forecast Horizon",
    y = "Probability of Rain"
  ) +
  theme_minimal()



library(dplyr)
library(tidyr)

forecast_table <- forecast_7days %>%
  select(Day, Location, Predicted_Probability) %>%
  pivot_wider(
    names_from = Location,
    values_from = Predicted_Probability
  )

forecast_table

forecast_table <- forecast_7days %>%
  mutate(
    Predicted_Probability =
      round(Predicted_Probability * 100, 2)
  ) %>%
  select(Day, Location, Predicted_Probability) %>%
  pivot_wider(
    names_from = Location,
    values_from = Predicted_Probability
  )

forecast_table


library(knitr)
library(kableExtra)

forecast_table %>%
  kable(
    caption = "7-Day Conditional Forecast of Rain Probability (%)",
    digits = 2
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover"),
    full_width = FALSE
  )





summary(logit_model)

# Odds Ratios
odds_ratios <- exp(coef(logit_model))

# Confidence Intervals
conf_int <- exp(confint(logit_model))

# P-values
p_values <- summary(logit_model)$coefficients[,4]

# Final table
or_table <- data.frame(
  Variable = names(odds_ratios),
  Odds_Ratio = round(odds_ratios, 3),
  CI_Lower = round(conf_int[,1], 3),
  CI_Upper = round(conf_int[,2], 3),
  P_Value = round(p_values, 4)
)

or_table



or_table_report <- or_table %>%
  filter(!grepl("factor\\(Location\\)", Variable))

or_table_report


library(knitr)
library(kableExtra)

or_table_report %>%
  kable(
    caption = "Odds Ratios from the Logit Model",
    digits = 3
  ) %>%
  kable_styling(
    bootstrap_options = c("striped","hover"),
    full_width = FALSE
  )