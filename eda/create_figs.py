import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path



# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

df = pd.read_csv(r"data/cleaned_data/master_data.csv")

grouped = df.groupby(["year", "combined_course"], as_index=False).agg({'num_students': 'sum', 'mean_student_price': 'mean'})
sns.lineplot(
    data=grouped,
    x="year",
    y="mean_student_price",   # or use enrollment-based column from master
    hue="combined_course",
    marker="o",
)
plt.savefig("eda/figs/price_trends.png", dpi=300, bbox_inches='tight')
plt.close()

sns.lineplot(
    data=grouped,
    x="year",
    y="num_students",   # or use enrollment-based column from master
    hue="combined_course",
    marker="o",
)
plt.savefig("eda/figs/enrollment_trends.png", dpi=300, bbox_inches='tight')
plt.close()