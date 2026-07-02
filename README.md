# Bathroom Renovation - Management Dashboard

## Files
- `clean_data.py` — minimal cleaning script. Run once: `python clean_data.py`
  (reads `raw_data.xlsx`, writes `cleaned_data.csv` + `cleaned_data.xlsx`)
- `cleaned_data.csv` — already-cleaned dataset (pre-generated, ready to use)
- `app.py` — the Streamlit dashboard
- `requirements.txt` — dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repo (include cleaned_data.csv, app.py, requirements.txt)
2. Go to share.streamlit.io -> New app -> point to app.py in your repo
3. Deploy

## Key data notes
- Grain: 1 row = 1 bathroom (Project Child Code). A customer (Project Parent Code)
  can have multiple bathroom rows.
- Revenue figures use a customer-level de-duplication rule: if a customer's
  Quotation Value / Total Revenue Collected is identical across all their bathroom
  rows (a project-level total copy-pasted, found in 51/256 multi-bathroom
  customers), it is counted once, not summed per row.
