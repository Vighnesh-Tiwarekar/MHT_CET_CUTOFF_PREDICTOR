import pandas as pd

def combine_college_data(cutoff_file, intake_file, output_file):
    print("Reading files...")
    # Load CSVs - using low_memory=False to handle mixed types in large datasets
    df_cutoff = pd.read_csv(cutoff_file, low_memory=False)
    df_intake = pd.read_csv(intake_file, low_memory=False)

    # 1. Clean Column Names
    # This removes any hidden spaces or newlines from the headers
    df_cutoff.columns = df_cutoff.columns.str.strip()
    df_intake.columns = df_intake.columns.str.strip()

    # 2. Clean 'Branch Code' and 'Year'
    # We convert to string and strip '.0' to ensure '1E+08' or '1002.0' matches '1002'
    for df in [df_cutoff, df_intake]:
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        
        if 'Branch Code' in df.columns:
            df['Branch Code'] = df['Branch Code'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    # 3. Drop duplicate intake entries
    # Ensure we only have one row per Year per Branch Code in the seat matrix
    df_intake = df_intake.drop_duplicates(subset=['Year', 'Branch Code'])

    # 4. Merge the data
    # We perform a left join on the Cutoff data
    print("Merging datasets on Year and Branch Code...")
    merged_df = pd.merge(
        df_cutoff, 
        df_intake[['Year', 'Branch Code', 'Seat_Intake']], 
        on=['Year', 'Branch Code'], 
        how='left'
    )

    # 5. Save the result
    merged_df.to_csv(output_file, index=False)
    print(f"--- Process Complete ---")
    print(f"Combined data saved to: {output_file}")
    print(f"Total rows processed: {len(merged_df)}")

if __name__ == "__main__":
    combine_college_data(
        cutoff_file='Master_Cutoff_Data.csv', 
        intake_file='Master_Seat_Matrix.csv', 
        output_file='Cutoff_Seat_Combo.csv'
    )