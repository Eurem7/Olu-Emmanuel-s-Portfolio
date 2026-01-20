📊 Bank Marketing Campaign Analysis (Power BI & Python)
🔍 Overview

This project analyzes a bank’s direct marketing campaign data to understand customer conversion behavior, campaign effectiveness, and return on investment (ROI).
The goal is to identify which contact strategies drive the highest conversions and profitability and provide actionable recommendations for optimizing future campaigns.

The analysis combines Python (EDA & feature engineering) with an interactive Power BI dashboard for business storytelling.

📁 Dataset

Records: 45,211 customers

Features: 17 (demographic, campaign, and economic variables)

Target Variable:

y → Whether the customer subscribed (yes / no)

Data Characteristics

Overall conversion rate: ~11.7%

Strong class imbalance (expected in marketing campaigns)

No missing values

Categorical "unknown" values retained to avoid biasing results

🧹 Data Preparation

Minimal preprocessing was applied to preserve data integrity:

Encoded target variable (converted)

Validated numeric and categorical features

Retained "unknown" categories as valid values

Feature Engineering

Campaign Intensity Groups

1 Contact

2–3 Contacts

4+ Contacts

This grouping enables fair comparison while reducing distortion from extreme contact counts.

📈 Key Insights
1️⃣ Conversion Rate by Campaign Intensity
Campaign Intensity	Conversion Rate
1 Contact	~14.6%
2–3 Contacts	~11.2%
4+ Contacts	~7.35%

Conversion declines as contact frequency increases

Conversion uplift (best vs worst): ~98.5%

📌 Insight: Over-contacting customers reduces effectiveness.

2️⃣ Conversion Rate by Contact Method
Contact Method	Conversion Rate
Cellular	~14.9%
Telephone	~13.4%
Unknown	~4.1%

📌 Insight: Cellular campaigns outperform other channels; unknown contacts are largely ineffective.

3️⃣ ROI Analysis

Assumptions:

Revenue per conversion: $50

Cost per contact: $1

ROI by Contact Method

Cellular: ~646%

Telephone: ~571%

Unknown: ~104%

📌 Insight: Cellular campaigns deliver the highest profitability.

ROI by Campaign Intensity

1 Contact: Highest ROI

2–3 Contacts: Moderate ROI

4+ Contacts: Lowest ROI

📌 Insight: Both efficiency and profitability decrease with increased contact frequency.

📊 Power BI Dashboard

The interactive dashboard includes:

KPI Cards

Overall Revenue

Total Customers

Total Conversions

Overall Conversion Rate

Total Cost

Overall ROI

Performance Visuals

Conversions by Campaign Intensity

Conversions by Contact Method

ROI by Campaign Intensity

ROI by Contact Method

Distribution Visuals

Customer Count by Campaign Intensity

Customer Distribution by Contact Method

📌 Designed for executive-level decision making with clear performance comparisons.

🎨 Design Choices

Green: High performance / profitability

Blue: Moderate performance

Grey: Low performance

Neutral background for clarity and readability

Consistent % and currency formatting

💡 Business Recommendations

Prioritize single-contact campaigns

Focus marketing resources on cellular channels

Reduce excessive follow-ups to avoid diminishing returns

Limit spending on unknown contact methods

🛠 Tech Stack

Python: pandas (EDA & feature engineering)

Power BI: DAX, modeling, visualization

CSV: Data storage

📌 Conclusion

This project demonstrates how data analysis and visualization can translate raw marketing data into clear, actionable business insights.
The findings support smarter campaign design, reduced costs, and improved ROI.