from datetime import datetime
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Lassa Fever Dashboard API",
    description="Aggregated outbreak data for public health monitoring",
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data file path
DATA_FILE = os.getenv("DATA_FILE", "extended_patient_outbreak_dataset_5000_diverse.csv")


def load_and_validate_data() -> pd.DataFrame:
    """Load and validate the patient dataset."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    required_columns = {
        'Patient_ID', 'Age', 'Sex', 'Outcome', 'State', 'LGA',
        'Case_Status', 'Last_Update'
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


@app.get("/")
def root():
    return {
        "message": "Lassa Fever Dashboard API is live!",
        "endpoints": ["/summary"]
    }


@app.get("/summary")
def get_summary():
    """
    Returns aggregated outbreak summary in dashboard-ready JSON format.
    Includes KPIs, demographics, LGA breakdown, and WEEKLY trends.
    """
    try:
        df = load_and_validate_data()
        df = df.copy()

        # --- Convert Last_Update to datetime and extract Year & ISO Week ---
        df['Last_Update'] = pd.to_datetime(df['Last_Update'], errors='coerce')
        df['Year'] = df['Last_Update'].dt.year
        df['Week'] = df['Last_Update'].dt.isocalendar().week  # ISO week (1-53)
        df['YearWeek'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

        # --- Age Grouping ---
        def get_age_group(age):
            if pd.isna(age):
                return "Other/Unknown"
            if age <= 14:
                return "0-14"
            elif age <= 49:
                return "15-49"
            else:
                return "50+"
        
        df['AgeGroup'] = df['Age'].apply(get_age_group)

        # --- Gender Normalization ---
        df['Gender'] = (
            df['Sex']
            .astype(str)
            .str.strip()
            .str.title()
            .replace({
                'M': 'Male',
                'F': 'Female',
                'Male.': 'Male',
                'Female.': 'Female',
                'nan': 'Other/Unknown'
            })
            .fillna("Other/Unknown")
        )

        # --- Core Metrics ---
        total_cases = len(df)
        confirmed = df[df['Case_Status'] == 'Confirmed']
        confirmed_cases = len(confirmed)

        deaths = confirmed[confirmed['Outcome'] == 'Deceased'].shape[0]
        recoveries = confirmed[confirmed['Outcome'] == 'Discharged'].shape[0]

        fatality_rate = round(deaths / confirmed_cases * 100, 1) if confirmed_cases > 0 else 0.0
        recovery_rate = round(recoveries / confirmed_cases * 100, 1) if confirmed_cases > 0 else 0.0

        states_affected = df['State'].nunique()
        lgas_affected = df['LGA'].nunique()

        # --- Demographics Breakdown ---
        def build_breakdown(series, total):
            counts = series.value_counts()
            return [
                {
                    "category": str(idx),
                    "count": int(count),
                    "percentage": round(count / total * 100, 1)
                }
                for idx, count in counts.items()
            ]

        age_breakdown = build_breakdown(df['AgeGroup'], total_cases)
        gender_breakdown = build_breakdown(df['Gender'], total_cases)

        # --- LGA Breakdown (with Year) ---
        lga_agg = df.groupby(['LGA', 'State'], dropna=False).agg(
            cases=('Patient_ID', 'count'),
            deaths=('Outcome', lambda x: (x == 'Deceased').sum()),
            recoveries=('Outcome', lambda x: (x == 'Discharged').sum()),
            last_update=('Last_Update', 'max'),
            year=('Year', 'max')
        ).reset_index()

        lga_summary = []
        for _, row in lga_agg.iterrows():
            cases = row['cases']
            recovery_pct = round(row['recoveries'] / cases * 100) if cases > 0 else 0
            lga_summary.append({
                "lga": str(row['LGA']) if pd.notna(row['LGA']) else "Unknown",
                "state": str(row['State']) if pd.notna(row['State']) else "Unknown",
                "cases": int(cases),
                "deaths": int(row['deaths']),
                "recovery_rate_percent": int(recovery_pct),
                "last_update": str(row['last_update']) if pd.notna(row['last_update']) else None,
                "year": int(row['year']) if pd.notna(row['year']) else None
            })

        # --- WEEKLY TREND AGGREGATION (NEW) ---
        weekly_trend = df.groupby('YearWeek').agg(
            total_cases=('Patient_ID', 'count'),
            confirmed_cases=('Case_Status', lambda x: (x == 'Confirmed').sum()),
            deaths=('Outcome', lambda x: (x == 'Deceased').sum()),
            recoveries=('Outcome', lambda x: (x == 'Discharged').sum())
        ).reset_index()

        # Sort by YearWeek (format YYYY-Www ensures correct order)
        weekly_trend = weekly_trend.sort_values('YearWeek').reset_index(drop=True)

        weekly_summary = []
        for _, row in weekly_trend.iterrows():
            weekly_summary.append({
                "year_week": row['YearWeek'],
                "total_cases": int(row['total_cases']),
                "confirmed_cases": int(row['confirmed_cases']),
                "deaths": int(row['deaths']),
                "recoveries": int(row['recoveries'])
            })

        # --- Final JSON Response ---
        return {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "data_file": DATA_FILE,
                "total_records": total_cases
            },
            "kpi": {
                "confirmed_cases": confirmed_cases,
                "recoveries": recoveries,
                "deaths": deaths,
                "fatality_rate_percent": fatality_rate,
                "recovery_rate_percent": recovery_rate,
                "states_affected": states_affected,
                "lgas_affected": lgas_affected
            },
            "demographics": {
                "age_groups": age_breakdown,
                "gender": gender_breakdown
            },
            "lga_breakdown": lga_summary,
            "weekly_trend": weekly_summary  #Weekly data for dashboard charts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")
