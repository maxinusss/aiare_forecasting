import os
from pathlib import Path
import pandas as pd

# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)


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

        required_cols = {"Course Type Name", "Start Date", "Enrolled"}
        if not required_cols.issubset(df.columns):
            # Skip files that do not look like the expected format
            continue

        temp = df[["Course Type Name", "Start Date", "Enrolled"]].copy()
        temp["Start Date"] = pd.to_datetime(temp["Start Date"], errors="coerce")
        temp["Enrolled"] = pd.to_numeric(temp["Enrolled"], errors="coerce").fillna(0)

        temp = temp.dropna(subset=["Start Date", "Course Type Name"])

        temp["month"] = temp["Start Date"].dt.month
        temp["year"] = temp["Start Date"].dt.year
        temp["course"] = temp["Course Type Name"]

        frames.append(temp[["month", "year", "course", "Enrolled"]])

    if not frames:
        raise ValueError(
            "No CSV files in the folder matched the expected columns: "
            "'Course Type Name', 'Start Date', 'Enrolled'"
        )

    combined = pd.concat(frames, ignore_index=True)

    result = (
        combined.groupby(["month", "year", "course"], as_index=False)["Enrolled"]
        .sum()
        .rename(columns={"Enrolled": "enrolled"})
        .sort_values(["year", "month", "course"])
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

        temp["month"] = temp["Course Start Date"].dt.month
        temp["year"] = temp["Course Start Date"].dt.year
        temp["course"] = temp["Course Type Name"]
        temp["state"] = temp["Course State"]
        temp["course_mode_of_travel"] = temp["Course Mode of Travel"]

        frames.append(
            temp[
                [
                    "month",
                    "year",
                    "course",
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
            ["month", "year", "course", "state", "course_mode_of_travel"],
            as_index=False,
        )["ID"]
        .count()
        .rename(columns={"ID": "id_count"})
        .sort_values(
            ["year", "month", "course", "state", "course_mode_of_travel"]
        )
        .reset_index(drop=True)
    )

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    return result


#combine course and student information at correct level
courses = combine_course_enrollment(
    "data/raw_data/Courses", "data/cleaned_data/course_enrollment.csv"
)
students = combine_student_counts(
    "data/raw_data/Students", "data/cleaned_data/student_enrollment.csv"
)