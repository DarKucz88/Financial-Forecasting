############################################################

# FORECASTING AND SIMULATIONS PROJECT

# Conditional Ex-Ante Forecast of RainTomorrow

# Cities: Sydney, Perth, Darwin, Hobart, Alice Springs

############################################################

#-----------------------------------------------------------

# 1. Select representative Australian locations

#-----------------------------------------------------------

# East, West, North, South and Central Australia

selected_cities <- c("Sydney", "Perth", "Darwin", "Hobart", "AliceSprings")

#-----------------------------------------------------------

# 2. Load training and test datasets

#-----------------------------------------------------------

train <- read.csv("Weather-Training-Data.csv")
test  <- read.csv("Weather-Test-Data.csv")

# Verify that all selected locations exist in both datasets

unique(train$Location)

selected_cities %in% unique(train$Location)
selected_cities %in% unique(test$Location)

#-----------------------------------------------------------

# 3. Filter datasets to selected locations only

#-----------------------------------------------------------

train_cities <- subset(train, Location %in% selected_cities)
test_cities  <- subset(test,  Location %in% selected_cities)

#-----------------------------------------------------------

# 4. Prepare binary variables

#-----------------------------------------------------------

# Convert RainToday from Yes/No to 1/0

train_cities$RainToday <- ifelse(train_cities$RainToday == "Yes", 1, 0)
test_cities$RainToday  <- ifelse(test_cities$RainToday == "Yes", 1, 0)

# Convert dependent variable to numeric format

train_cities$RainTomorrow <- as.numeric(train_cities$RainTomorrow)

#-----------------------------------------------------------

# 5. Select variables used in the Logit model

#-----------------------------------------------------------

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

# Remove observations with missing values

train_model <- na.omit(train_model)

# Variables available in future (test) data

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

#-----------------------------------------------------------

# 6. Estimate Logit model with location fixed effects

#-----------------------------------------------------------

logit_model <- glm(
  RainTomorrow ~ Rainfall +
    Sunshine +
    Humidity3pm +
    Pressure3pm +
    Cloud3pm +
    WindGustSpeed +
    Temp3pm +
    RainToday +
    factor(Location),
  data = train_model,
  family = binomial(link = "logit")
)

# Display estimation results

summary(logit_model)

#-----------------------------------------------------------

# 7. Generate predicted probabilities

#-----------------------------------------------------------

test_model_clean <- na.omit(test_model)

test_model_clean$Predicted_Probability <- predict(
  logit_model,
  newdata = test_model_clean,
  type = "response"
)

#-----------------------------------------------------------

# 8. Create 7-day conditional forecast

#-----------------------------------------------------------

library(dplyr)

forecast_7days <- test_model_clean %>%
  group_by(Location) %>%
  slice(1:7) %>%
  mutate(Day = paste0("Day ", row_number())) %>%
  ungroup()

#-----------------------------------------------------------

# 9. Create forecast table

#-----------------------------------------------------------

library(tidyr)

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

#-----------------------------------------------------------

# 10. Visualise 7-day rainfall probabilities

#-----------------------------------------------------------

library(ggplot2)

ggplot(
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
  labs(
    title = "Conditional Ex-Ante Forecast Across Australia",
    subtitle = "Probability of Rain Tomorrow, 7-Day Horizon",
    x = "Forecast Horizon",
    y = "Probability of Rain"
  ) +
  theme_minimal()

#-----------------------------------------------------------

# 11. Export professional forecast table

#-----------------------------------------------------------

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

############################################################

# END OF CONDITIONAL FORECAST ANALYSIS

############################################################
