# Tennis Edge AI — rebuilt clean project

This project uses Jeff Sackmann's historical ATP and WTA match-results data via the archival mirror `Aneeshers/tennis-sackmann-archive`, which preserves the original data attribution and license.

## Files
- `train.py`: automatically downloads yearly ATP/WTA match CSVs (2000 through current year), builds pre-match overall Elo, surface Elo, ranking, recent form and experience features, trains an XGBoost + logistic-regression ensemble, performs a chronological holdout evaluation, and saves model artifacts.
- `app.py`: Streamlit match-winner and odds-value dashboard. It uses exactly the same seven feature names and order as `train.py`.
- `requirements.txt`: dependencies.

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training requires internet access and can take several minutes. The script skips unavailable year files and reuses already downloaded CSVs.

## Data source and license
Archive: https://github.com/Aneeshers/tennis-sackmann-archive
Original ATP: https://github.com/JeffSackmann/tennis_atp
Original WTA: https://github.com/JeffSackmann/tennis_wta

The historical data is distributed under CC BY-NC-SA 4.0: attribution required, non-commercial use only, and share-alike for adaptations/redistribution. This is suitable for personal, non-commercial research. Do not use it in a commercial betting product without permission/a suitable license.

## Limitations
This is a research prototype, not a proven profitable betting system. It does not fetch live odds, current injuries, withdrawals, or today's fixtures. The player state is derived from the downloaded historical match rows and can be stale. Validate calibration and performance on later unseen matches before relying on probabilities.
