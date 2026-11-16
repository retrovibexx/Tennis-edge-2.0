# Tennis Edge AI

A Streamlit tennis match research dashboard using historical ATP/WTA match data.

## First launch

The app automatically downloads available historical match CSVs, builds player ratings and recent-form features, trains its models, and saves the model artifacts. **There is no manual `python train.py` step and the app does not stop with a “model not trained yet” message.**

On Streamlit Community Cloud, set the app's main file to `app.py`. The app calls `train.main()` automatically when the model artifacts are missing.

## Files

- `app.py` — Streamlit interface and automatic first-launch training
- `train.py` — data download, feature generation, model training, and validation
- `requirements.txt` — Python dependencies
- `README.md` — setup notes

## Data and limitations

Historical match data is sourced from Jeff Sackmann's tennis datasets and a public archive mirror. Review the upstream repositories and their licenses/attribution requirements before redistribution or commercial use. Historical data does not include a complete live feed of injuries, withdrawals, weather, or current odds. Predictions are estimates, not guarantees or betting advice.

## Deployment

1. Push these files to the repository's `main` branch.
2. In Streamlit Community Cloud, deploy or reboot the app with `app.py` as the entry point.
3. On the first run, allow time for the historical data download and model training to finish. Later app reruns reuse the trained artifacts while the app runtime remains available.
