import pandas as pd

def add_bmi_category(df):
    bmi_bins = [0, 18.5, 24.9, 29.9, 100]
    category_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df = df.copy()  
    if 'BMI' not in df.columns:
        raise ValueError("Dataframe must contain a 'BMI' column.")
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=category_labels, include_lowest=True)
    return df


