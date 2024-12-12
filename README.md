# Random Forest Model (BCG X Data Science Project Part 3)
![Introductory Picture](Random_Forest.png)
## Introduction
This is Part 3 of a project from the [BCG X Data Science micro-internship](https://www.theforage.com/simulations/bcg/data-science-ccdz). The Boston Consulting Group (BCG) is an American global consulting firm that  partners with leaders in business and society to tackle their most important challenges. It is one of the world's 3 largest consulting firms along with McKinsey & Company and Bain & Company. BCG X is a new initiative from BCG that combines the firm's consulting expertise with tech building and design.

In this task, I take on the role of a junior data analyst employed at BCG X. BCG X's client, a major gas and electricity utility called PowerCo, is concerned about their customers leaving for better offers from other energy providers. **In this part of the project, I will create a Random Forest model using the data we have created in part 2. The Random Forest model will predict which customers will leaver PowerCo and will determine which features are the most influential in determining whether a customer will leave. We will also determine if our hypothesis, that price sensitivity is an important factor for churn, is correct.**

## Problem Statement
PowerCo has expressed concern over their customers leaving them for better offers from competing energy companies. This concern is exacerbated by the fact that the energy market has had a lot of change in recent years and there are more options than ever for customers to choose from. During a meeting with the Associate Director of the Data Science team, **one potential reason for churn is price sensitivity.** I am tasked with investigating this hypothesis. **We will use the Random Forest model to determine what features are most influential to customer churn. The data we have created in part 2 will be used as input to train and test the model.**

## Skills Demonstrated
* Python
* Machine Learning
* Random Forest
* Training and Testing Machine Learning Models
* Data Manipulation
* Data Visualization

## Data Sourcing
This original data was provided to me by the BCG X Data Science microinternship hosted by Forage. In part 2, feature engineering was conducted on the data. A copy of the altered data is included in this repository under the file name: data_for_predictions.csv.

## Data Attributes
The data contains information about power consumption, sales channels, forecasted power consumption, and whether the client has churned or not. Each row contains data for 1 client.

These attributes are included in the original data:
* id - Client company identifier.
* cons_12m - Electricity consumption of the past 12 months.
* cons_gas_12m - Gas consumption of the past 12 months.
* cons_last_month - Electricity consumption of the last month.
* forecast_cons_12m - FForecasted electricity consumption for next 12 months.
* forecast_discount_energy - Forecasted value of current discount.
* forecast_meter_rent_12m - Forecasted bill of meter rental for the next 2 months.
* forecast_price_energy_off_peak - Forecasted energy price for 1st period (off peak).
* forecast_price_energy_peak - Forecasted energy price for 2nd period (peak).
* forecast_price_pow_off_peak - Forecasted power price for 1st period (off peak).
* has_gas - Indicated if client is also a gas client.
* imp_cons - Current paid consumption.
* margin_net_pow_ele - Net margin on power subscription.
* nb_prod_act - Number of active products and services.
* net_margin - Total net margin.
* pow_max - Subscribed power.
* churn - Has the client churned over the next 3 months.

These attributes are added to the data from part 2:
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
