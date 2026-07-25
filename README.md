# Accredian_Assesment

# Instructor Effectiveness Modeling

## Overview
This project analyzes batch-level learner outcome, engagement, and feedback data 
from an EdTech platform to define an Instructor Effectiveness Score and predict 
effectiveness tiers (Low / Medium / High) using a machine learning model.

## Dataset
- `instructor_effectiveness_dataset_2000_rows.csv`
- 2000 batch-level records across 120 instructors and 25 courses
- Includes learner outcome metrics (completion rate, dropout rate, score improvement, 
  quiz score), engagement metrics (watch time, assignment submission, forum activity), 
  and feedback metrics (feedback score, feedback response rate)

## Approach
1. **Exploratory Data Analysis** — examined distributions, missing values, and 
   correlations between metrics.
2. **Effectiveness Score Definition** — defined effectiveness as a weighted 
   combination of outcomes (50%), engagement (30%), and feedback (20%), with all 
   metrics min-max scaled before combining.
3. **Aggregation** — batch-level records aggregated to instructor level using mean 
   values across all batches taught.
4. **Modeling** — trained a Random Forest classifier to predict effectiveness tier 
   from instructor-level features.
5. **Evaluation** — assessed using accuracy, precision/recall, F1-score, and a 
   confusion matrix.
6. **Interpretation** — analyzed feature importances and discussed limitations, 
   confounds, and real-world applicability.

## Results
- Model accuracy: ~92%
- Most influential features: `completion_rate`, `dropout_rate`, `feedback_response_rate`
- Full interpretation and analysis questions are answered in the notebook

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `Accredian_assesment.ipynb` in Jupyter and run all cells

## Limitations
This model does not account for confounding factors such as course difficulty, 
student cohort quality, or instructor experience. It should be treated as a 
decision-support tool rather than a standalone instructor evaluation mechanism.
