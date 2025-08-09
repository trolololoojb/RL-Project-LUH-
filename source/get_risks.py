import pandas as pd

# Calculate risk score for our selected features based on mabilton/fremtpl2 dataset
# https://huggingface.co/datasets/mabilton/fremtpl2
df = pd.read_csv("hf://datasets/mabilton/fremtpl2/freMTPL2freq.csv")
df['claim_occured'] = (df['ClaimNb'] > 0).astype(int)

MIN_AGE = df['DrivAge'].min()
MAX_AGE = df['DrivAge'].max()

bins = [MIN_AGE-1, 25, 35, 45, 55, 65, 75, 85, MAX_AGE]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-100']
df['age_group'] = pd.cut(df['DrivAge'], bins=bins, labels=labels)

prob_per_age_group = df.groupby('age_group')['claim_occured'].mean().to_dict()

prob_per_area = df.groupby('Area')['claim_occured'].mean()
areas_order = ['A', 'B', 'C', 'D', 'E', 'F']
prob_area_list = [prob_per_area.get(area, 0) for area in areas_order]

def get_base_risk():
    return len(df[df['claim_occured'] > 0]) / len(df)

def get_region_risk(region_index):
    if 0 <= region_index < len(prob_area_list):
        return prob_area_list[region_index]
    else:
        raise ValueError("Region index muss zwischen 0 und 5 liegen.")

def get_age_risk(age):
    age_group = pd.cut([age], bins=bins, labels=labels)[0]
    return prob_per_age_group.get(age_group, 0)