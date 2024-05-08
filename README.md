# Air Condition Prediction (PM10)

### Problem Addressed with Machine Learning
The project focuses on predicting PM10 particle concentration levels in the air. Accurate forecasting of PM10 is crucial for public health, especially in urban areas, as it enables local authorities and residents to take preventive actions in a timely manner.

### Historical Data Utilized
The model will be trained on historical data from 2019-2022, including daily measurements of PM10 concentration. This data may be supplemented with meteorological conditions such as temperature, humidity, and wind speed, which can influence the distribution and accumulation of PM10 particles.

### Data Analysis and Model Calculations
The analysis will focus on time-series data of PM10 concentrations, potentially enriched with meteorological and other pollution data. The model will calculate predicted PM10 levels for future days based on sequences of previous measurements and weather conditions.

### Type of Model Applied
A Gated Recurrent Unit (GRU) model will be used. GRU is effective for time-series data due to its ability to handle long sequences without running into vanishing gradient problems, making it suitable for predicting patterns and trends in air quality data.
