# Data Scientist Assessment: Time Series Forecasting Challenge

## Dataset Description

This dataset contains electricity consumption data from 370 individual meters collected over 4 years at 15-minute intervals. The data represents real-world energy usage patterns and is ideal for time series analysis and forecasting tasks.

### Data Structure
- **Timestamp Column:** Date and time in format "YYYY-MM-DD HH:MM:SS"
- **Meter Columns:** 370 columns labeled MT_001 through MT_370
- **Values:** Electricity consumption readings (numeric, with decimal precision)
- **File Format:** Semicolon-separated values (CSV with ';' delimiter)
- **Missing Data:** Some meters may have zero or missing readings

### Sample Data Preview
```
"";"MT_001";"MT_002";"MT_003"...;"MT_370"
"2011-01-01 00:15:00";0;0;0;...;0
"2011-01-01 00:30:00";0;0;0;...;0
```

## Assessment Task

### Objective
Develop a time series forecasting model to predict electricity consumption for the next 30 days using historical data aggregated to daily intervals.

### Core Requirements

#### 1. Data Preprocessing & Exploration 

#### 2. Feature Engineering 

#### 3. Model Development 

#### 4. Model Evaluation & Validation 

#### 5. Forecasting & Visualization 

## Technical Specifications

### Data Processing Requirements
- Aggregate 15-minute data to daily totals/averages
- Handle timezone considerations if applicable
- Ensure data quality and consistency across the time series

### Model Requirements
- Forecast horizon: 30 days into the future
- Provide point forecasts and prediction intervals
- Handle multiple time series appropriately
- Consider computational efficiency for production deployment

### Deliverables Expected

#### 1. Jupyter Notebook/Python Script
- Well-documented code with clear explanations
- Reproducible analysis with proper random seeds
- Efficient and clean implementation
