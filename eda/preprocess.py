import os
from pathlib import Path
import pandas as pd
from utils import merge_dataframes_on_keys

# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

# Date filter: only include data after this date
FILTER_DATE = pd.Timestamp("2017-07-01")


def combine_course_enrollment(folder_path: str, output_path: str) -> pd.DataFrame:
    """
    Read all CSV files in a folder that match the course-export shape,
    and return a combined dataset with:
        month, year, course, enrolled

    Assumptions based on the sample CSV:
    - course column is "Course Type Name"
    - date column is "Start Date"
    - enrolled column is "Enrolled"
    
    Args:
        folder_path: Path to folder containing CSV files
        output_path: Path to save the combined CSV file
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    frames = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required_cols = {"Course Type Name", "Start Date", "Enrolled", "Student Price"}
        if not required_cols.issubset(df.columns):
            # Skip files that do not look like the expected format
            continue

        temp = df[["Course Type Name", "Start Date", "Enrolled", "Student Price"]].copy()
        temp["Start Date"] = pd.to_datetime(temp["Start Date"], errors="coerce")
        temp["Enrolled"] = pd.to_numeric(temp["Enrolled"], errors="coerce").fillna(0)

        # Normalize currency values like "$390.00" to numeric
        temp["Student Price"] = (
            temp["Student Price"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", pd.NA)
        )
        temp["Student Price"] = pd.to_numeric(temp["Student Price"], errors="coerce")

        temp = temp.dropna(subset=["Start Date", "Course Type Name"])
        
        # Filter to dates after FILTER_DATE
        temp = temp[temp["Start Date"] > FILTER_DATE]

        temp["month"] = temp["Start Date"].dt.month
        temp["year"] = temp["Start Date"].dt.year
        temp["course"] = temp["Course Type Name"]

        # Create combined_course for grouping while preserving original course text
        temp["combined_course"] = temp["course"].str.strip().str.lower()
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*1|aiare\s*1", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 1"
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*2|aiare\s*2", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 2"
        # Do not remap non-AIARE rescue courses; keep them as-is in combined_course

        frames.append(temp[["month","year","course","combined_course","Enrolled","Student Price"]])

    if not frames:
        raise ValueError(
            "No CSV files in the folder matched the expected columns: "
            "'Course Type Name', 'Start Date', 'Enrolled'"
        )

    combined = pd.concat(frames, ignore_index=True)

    result = (
        combined.groupby(["month", "year", "combined_course"], as_index=False)
        .agg({"Enrolled": "sum", "Student Price": "mean", "course": "first"})
        .rename(columns={"Enrolled": "enrolled", "Student Price": "student_price"})
        .sort_values(["year", "month", "combined_course"])
        .reset_index(drop=True)
    )

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    return result

def combine_student_counts(folder_path: str, output_path: str) -> pd.DataFrame:
    """
    Read all CSV files in a folder that match the student-export shape,
    and return a combined dataset with:
        month, year, course, state, course_mode_of_travel, id_count

    Assumptions based on the sample CSV:
    - ID column is "ID"
    - course column is "Course Type Name"
    - state column is "Course State"
    - travel mode column is "Course Mode of Travel"
    - date column is "Course Start Date"
    
    Args:
        folder_path: Path to folder containing CSV files
        output_path: Path to save the combined CSV file
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    required_cols = {
        "ID",
        "Course Type Name",
        "Course State",
        "Course Mode of Travel",
        "Course Start Date",
    }

    frames = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if not required_cols.issubset(df.columns):
            # Skip files that do not match the expected layout
            continue

        temp = df[
            [
                "ID",
                "Course Type Name",
                "Course State",
                "Course Mode of Travel",
                "Course Start Date",
            ]
        ].copy()

        temp["Course Start Date"] = pd.to_datetime(
            temp["Course Start Date"], errors="coerce"
        )

        temp = temp.dropna(subset=["Course Start Date", "Course Type Name", "ID"])
        
        # Filter to dates after FILTER_DATE
        temp = temp[temp["Course Start Date"] > FILTER_DATE]

        temp["month"] = temp["Course Start Date"].dt.month
        temp["year"] = temp["Course Start Date"].dt.year
        temp["course"] = temp["Course Type Name"]
        temp["state"] = temp["Course State"]
        temp["course_mode_of_travel"] = temp["Course Mode of Travel"]

        # Create combined_course while keeping raw course text
        temp["combined_course"] = temp["course"].str.strip().str.lower()
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*1|aiare\s*1", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 1"
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*2|aiare\s*2", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 2"
        # Keep generic rescue courses untouched

        frames.append(
            temp[
                [
                    "month",
                    "year",
                    "course",
                    "combined_course",
                    "state",
                    "course_mode_of_travel",
                    "ID",
                ]
            ]
        )

    if not frames:
        raise ValueError(
            "No CSV files in the folder matched the expected columns: "
            "'ID', 'Course Type Name', 'Course State', "
            "'Course Mode of Travel', 'Course Start Date'"
        )

    combined = pd.concat(frames, ignore_index=True)

    result = (
        combined.groupby(
            ["month", "year", "combined_course"],
            as_index=False,
        ).agg({"ID":"count", "course":"first"})
        .rename(columns={"ID": "num_students"})
        .sort_values(
            ["year", "month", "combined_course"]
        )
        .reset_index(drop=True)
    )

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    return result

def combine_course_by_location_price(folder_path: str, output_path: str) -> pd.DataFrame:
    """Combine course CSVs, normalize price, and aggregate by location + price metrics."""
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required_cols = {"Course Type Name", "Start Date", "Enrolled", "Student Price", "Location"}
        if not required_cols.issubset(df.columns):
            continue

        temp = df[["Course Type Name", "Start Date", "Enrolled", "Student Price", "Location"]].copy()
        temp["Start Date"] = pd.to_datetime(temp["Start Date"], errors="coerce")
        temp = temp.dropna(subset=["Start Date", "Course Type Name", "Location"])
        temp = temp[temp["Start Date"] > FILTER_DATE]

        temp["year"] = temp["Start Date"].dt.year
        temp["month"] = temp["Start Date"].dt.month

        temp["Student Price"] = (
            temp["Student Price"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", pd.NA)
        )
        temp["Student Price"] = pd.to_numeric(temp["Student Price"], errors="coerce")

        temp["course"] = temp["Course Type Name"]
        temp["combined_course"] = temp["course"].str.strip().str.lower()
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*1|aiare\s*1", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 1"
        temp.loc[temp["combined_course"].str.contains(r"^aiare\s*2|aiare\s*2", regex=True) & temp["combined_course"].str.contains("rescue"), "combined_course"] = "aiare 2"

        frames.append(temp)

    if not frames:
        raise ValueError("No valid course files with required columns found.")

    combined = pd.concat(frames, ignore_index=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)

    return combined

def get_us_course_price_no_outliers(course_location_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Filter to US courses and remove price outliers."""
    # Derive country from Location if not already present: use last comma element
    if "Location" in course_location_df.columns:
        course_location_df["country"] = (
            course_location_df["Location"].astype(str)
            .str.split(",")
            .apply(lambda parts: parts[-1].strip() if len(parts) > 0 else "")
        )
    else:
        course_location_df["country"] = ""
    course_location_df = course_location_df[course_location_df["country"] == "United States"]  # filter to rows with valid country
    course_location_df["student_price"] = pd.to_numeric(course_location_df["Student Price"], errors="coerce")
    course_location_df = course_location_df.dropna(subset=["student_price", "month", "year", "combined_course"])  # ensure required

    # Compute 10th/90th percentile bounds within each combined_course group
    quantile_bounds = (
        course_location_df.groupby("combined_course")["student_price"]
        .quantile([0.10, 0.90])
        .unstack(level=-1)
        .rename(columns={0.10: "q10", 0.90: "q90"})
    )

    # Join bounds back to course_df
    course_location_df = course_location_df.merge(
        quantile_bounds[["q10", "q90"]],
        left_on="combined_course",
        right_index=True,
        how="left"
    )

    # Filter outliers using group-specific 10/90 bounds
    cleaned = course_location_df[
        (course_location_df["student_price"] >= course_location_df["q10"]) &
        (course_location_df["student_price"] <= course_location_df["q90"]) 
        ].copy()

    final = (
        cleaned.groupby(['year', 'month', 'combined_course'], as_index=False)
        ['student_price']
        .mean()
        .rename(columns={'student_price': 'mean_student_price'})
        .sort_values(['year', 'month', 'combined_course'])
    )
    final.to_csv(output_path, index=False)
    return final

#combine course and student information at correct level and only after 2017
courses = combine_course_enrollment(
    "data/raw_data/Courses", "data/cleaned_data/course_enrollment.csv"
).drop(columns=["course"])
students = combine_student_counts(
    "data/raw_data/Students", "data/cleaned_data/student_enrollment.csv"
)
course_location = combine_course_by_location_price(
    "data/raw_data/Courses", "data/cleaned_data/course_location_price.csv"
)
course_prices = get_us_course_price_no_outliers(
    course_location, "data/cleaned_data/course_price_us.csv")

master_data = merge_dataframes_on_keys(dfs=[courses, students, course_prices], keys=["month", "year", "combined_course"])
master_data.to_csv("data/cleaned_data/master_data.csv", index=False)