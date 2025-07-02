import pandas as pd
import numpy as np
import random
from faker import Faker

# Set seeds for reproducibility
fake = Faker()
np.random.seed(42)
random.seed(42)

# Number of records
n = 1000

# Age: skewed towards 18–24
age = np.random.normal(loc=22, scale=2.5, size=n).astype(int)
age = np.clip(age, 18, 30)

# Gender
gender = np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], size=n, p=[0.45, 0.45, 0.05, 0.05])

# Year of Study
year_of_study = np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year', 'Postgraduate'], size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])

# Monthly Disposable Income (in AED) with outliers
income = np.random.normal(loc=1500, scale=500, size=n).astype(int)
income[np.random.choice(n, size=10)] *= 3  # Injecting high-income outliers
income = np.clip(income, 100, 10000)

# Living situation
living = np.random.choice(['On-campus', 'Off-campus'], size=n, p=[0.4, 0.6])

# Transport mode
transport = np.random.choice(['Walk', 'Bike', 'Public Transport', 'Personal Vehicle', 'Ride-hailing'], size=n)

# Order frequency
order_freq = np.random.choice(['0', '1-2', '3-5', '6+'], size=n, p=[0.1, 0.4, 0.35, 0.15])

# Order time
order_time = np.random.choice(['Breakfast', 'Lunch', 'Dinner', 'Late night', 'Varies'], size=n)

# Preferred Cuisines (multi-label)
cuisine_options = ['Indian', 'Chinese', 'Italian', 'Fast Food', 'Healthy', 'Others']
cuisines = [', '.join(random.sample(cuisine_options, k=random.randint(1, 3))) for _ in range(n)]

# Choice factors (multi-label)
factor_options = ['Price', 'Speed', 'Quality', 'Variety', 'Sustainability', 'Reviews']
factors = [', '.join(random.sample(factor_options, k=random.randint(2, 3))) for _ in range(n)]

# Delay acceptance for freshness
fresh_food = np.random.choice(['Yes', 'No', 'Maybe'], size=n, p=[0.5, 0.3, 0.2])

# Uses food delivery apps
uses_apps = np.random.choice(['Yes', 'No'], size=n, p=[0.9, 0.1])

# Apps used (multi-label)
app_options = ['Zomato', 'Talabat', 'Uber Eats', 'Others']
apps = [', '.join(random.sample(app_options, k=random.randint(1, 3))) if use == 'Yes' else 'None' for use in uses_apps]

# Satisfaction score with current apps
satisfaction = np.random.randint(1, 6, size=n)

# Multi-restaurant interest
multi_restaurant = np.random.choice(['Yes', 'No', 'Maybe'], size=n, p=[0.6, 0.2, 0.2])

# Wait willingness
wait_fresh = np.random.choice(['Yes', 'No', 'Maybe'], size=n, p=[0.6, 0.2, 0.2])

# Willingness to pay for personalized healthy meal
willingness_to_pay = np.random.normal(loc=25, scale=10, size=n).astype(int)
willingness_to_pay = np.clip(willingness_to_pay, 10, 100)

# Personalized suggestions
personal_suggestions = np.random.choice(['Yes', 'No', 'Maybe'], size=n)

# Bundle deals
bundle_deals = np.random.choice(['Yes', 'No', 'Maybe'], size=n)

# Sustainability awareness
sustainability = np.random.choice(['Yes', 'No', 'Somewhat'], size=n)

# Empathy trained delivery preference
empathy_pref = np.random.choice(['Yes', 'No', 'Doesn’t matter'], size=n)

# Frustrations (free text samples)
frustration_samples = ['Late delivery', 'High prices', 'Cold food', 'No tracking', 'Limited menu']
frustrations = [random.choice(frustration_samples) for _ in range(n)]

# Common problems (multi-label)
problem_options = ['Late delivery', 'Cold food', 'High price', 'Limited variety', 'Poor packaging']
problems = [', '.join(random.sample(problem_options, k=random.randint(1, 3))) for _ in range(n)]

# Sustainable rewards
sustain_rewards = np.random.choice(['Yes', 'No', 'Maybe'], size=n)

# Loyalty scale
loyalty = np.random.randint(1, 6, size=n)

# Final DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Year_of_Study': year_of_study,
    'Monthly_Income_AED': income,
    'Living': living,
    'Transport_Mode': transport,
    'Order_Frequency': order_freq,
    'Order_Time': order_time,
    'Preferred_Cuisines': cuisines,
    'Choice_Factors': factors,
    'Accept_Delay_for_Freshness': fresh_food,
    'Uses_Apps': uses_apps,
    'Used_Apps': apps,
    'Satisfaction_Score': satisfaction,
    'Multi_Restaurant_Order': multi_restaurant,
    'Willing_to_Wait_for_Fresh_Food': wait_fresh,
    'Max_Willingness_to_Pay_AED': willingness_to_pay,
    'Wants_Personal_Suggestions': personal_suggestions,
    'Likes_Bundle_Deals': bundle_deals,
    'Cares_About_Sustainability': sustainability,
    'Empathy_Preference': empathy_pref,
    'Biggest_Frustration': frustrations,
    'Common_Problems': problems,
    'Sustainable_Rewards': sustain_rewards,
    'Loyalty_Likelihood': loyalty
})

# Export to CSV if needed
df.to_csv("synthetic_student_food_delivery.csv", index=False)

# Display preview
print(df.head())
