# COMP-2704: Supervised Machine Learning Term 
## Data Analysis and Preparation 
## 1) Problem and Use Case Definition
Several external and internal factors we face every day can affect our mental health condition either in a positive or a negative way. Despite our ability to tell what is good for our mental health apart from what is not, being aware of every single aspect and its consequences is not an easy task.

In this use case, we are going to leverage machine learning capabilities to identify positive and negative aspects of our lifes in detail, demonstrating how data science can enhance medical diagnosis by developing a classification model. The goal is to enable earlier detection and provide employees with timely and relevant treatments.

**Target Variable**: Mental_Health_Condition
- Class 0 (N): No mental health condition (52.4% of dataset)
- Class 1 (Y): Has mental health condition (47.6% of dataset)

**Objective**: Develop a supervised machine learning model to predict whether an individual has a mental health condition (Yes/No) based on demographic, lifestyle, and behavioral features.

### Dataset Description
The dataset is `mental_health_dataset`. This dataset comprises 50,000 records capturing various mental health and lifestyle factors. These factors are represented as the feature columns of the data set, which fall into the following categories:
- **Demographic:** This includes each person demographic information such as age, gender, occupation and country.
- **Mental Health Indicators:** This includes features like the stress level, consultation history and medication usage.
- **Lifestyle:** This includes information complementary information such as sleep hours, work hours, physical activity, social media usage and diet quality.
- **Additional Details:** This includes information about external factors such as smoking and alcohol consumption habits categorized into multiple levels.
### Classification Model Considerations.
**Problem Type**: Binary Classification

**Evaluation Focus**: We prioritize sensitivity (recall) to minimize false negatives, 
as failing to identify individuals with mental health conditions has serious consequences.