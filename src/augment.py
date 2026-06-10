import pandas as pd
import random
from src.config import PROCESSED_DATA_PATH

df = pd.read_csv(PROCESSED_DATA_PATH)

sides = ["left", "right", "bilateral"]
organs_map = {
    "left_kidney": "left kidney",
    "right_kidney": "right kidney",
    "liver": "liver",
    "spleen": "spleen"
} 

templates = [
    "The {organ_name} is {status} in this study.",
    "Findings are consistent with {status} {organ_name}.",
    "Visualization of the {organ_name} shows it is {status}.",
    "Clinical correlation suggests the {organ_name} is {status}."
]

augmented_data = []

for _, row in df.iterrows():
    
    augmented_data.append(row.to_dict())
 
 
    for _ in range(2):
        target_organ_col = random.choice(list(organs_map.keys()))
        status = row[target_organ_col]
        organ_name = organs_map[target_organ_col]
        
        new_text = random.choice(templates).format(organ_name=organ_name, status=status)
        
        new_row = row.to_dict()
        new_row["text"] = new_text
        augmented_data.append(new_row)

aug_df = pd.DataFrame(augmented_data)
aug_df.to_csv("Data/raw/augmented_reports.csv", index=False)
print("✅ Smart Augmentation done with medical templates.")