import os
from pathlib import Path
import pandas as pd
import numpy as np

# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

#-------DATA SETS--------#
master = pd.read_csv("data/cleaned_data/master_data.csv")
el_nino = pd.read_csv("data/cleaned_data/el_nino_la_nina_outlook_october.csv")
fred_data = pd.read_csv("data/cleaned_data/monthly_economic_features.csv")
#------------------------#


#-------COVID AND CMS FLAG--------#
# Build a date column from year/month when missing
if "date" not in master.columns:
    if "year" in master.columns and "month" in master.columns:
        master["date"] = pd.to_datetime(master.assign(day=1)[["year", "month", "day"]])
    else:
        raise ValueError("master_data.csv must have either 'date' or both 'year' and 'month' columns")
else:
    master["date"] = pd.to_datetime(master["date"])

# covid_flag for August 2020 through August 2021 inclusive
start_date = pd.Timestamp("2020-08-01")
end_date = pd.Timestamp("2021-08-31")
master["covid_flag"] = ((master["date"] >= start_date) & (master["date"] <= end_date)).astype(int)

#Add CMS loss flag for August 2023 and later
master['cms_loss_flag'] = ((master["date"] >= pd.Timestamp("2023-08-01"))).astype(int)
#-------------------------#


#-------EL NINO/LA NINA--------#
# Merge master (monthly) with el_nino (annual) by year
master["year"] = master["date"].dt.year
el_nino["year"] = el_nino["year"].astype(int)
merged = master.merge(el_nino[['year', 'enso_outlook']], on="year", how="left")

# If there are missing year values, do a forward/backward fill by year
merged = merged.sort_values(["date"])
merged[["enso_outlook"]] = merged[["enso_outlook"]].ffill().bfill()
#-------------------------------#


#-------ECONOMIC DATA--------#
fred_data=fred_data[['year', 'month', 'unemployment_rate', 'cpi', 'gas_price', 'economic_pressure_index']]
merged = merged.merge(fred_data, on=["year", "month"], how="left")
#----------------------------#


#-------SAVE OUT DATA --------#
merged.to_csv("data/cleaned_data/master_data_full.csv", index=False)

