# SportsBetting - Football Match Outcome Prediction

## Project Overview

This project builds a prediction model for football (soccer) match outcomes and compares predictions against betting odds to identify value bets. The goal is to generate probability estimates for match results (home win, draw, away win) that are more accurate than implied probabilities from bookmaker odds.

## Project Phases

### Phase 1: Data Capture (Scraping)
- Scrape historical match results, statistics, and betting odds from publicly available sources
- Target data: match results, team stats, player stats, head-to-head records, league standings, bookmaker odds
- Potential sources: football-data.co.uk, FBref, Transfermarkt, Understat, API-Football, WhoScored
- Store raw data in structured formats (CSV/Parquet) under `data/raw/`
- Build scrapers that are respectful of rate limits and robots.txt

### Phase 2: Research & Feature Engineering
- Explore which features are predictive of match outcomes (EDA)
- Research existing approaches: Elo ratings, Dixon-Coles model, Poisson models, xG-based models
- Engineer features: form, home/away strength, goal difference trends, expected goals, injuries, fatigue
- Document findings in notebooks under `notebooks/research/`

### Phase 3: Model Preparation & Training
- Implement baseline models (logistic regression, Poisson regression)
- Implement advanced models (gradient boosting, neural nets, ensemble methods)
- Train on historical data with proper temporal train/test splits (no future data leakage)
- Optimize hyperparameters via cross-validation
- Output: calibrated probability estimates for home/draw/away

### Phase 4: Testing & Evaluation
- Evaluate model accuracy: log-loss, Brier score, calibration plots, ROI simulation
- Compare model probabilities against bookmaker implied probabilities
- Backtest betting strategies (Kelly criterion, flat staking, threshold-based)
- Track performance on upcoming matches for live validation

## Project Structure

```
SportsBetting/
├── CLAUDE.md
├── data/
│   ├── raw/           # Raw scraped data
│   ├── processed/     # Cleaned and feature-engineered data
│   └── odds/          # Historical betting odds
├── scrapers/          # Web scraping scripts
├── notebooks/
│   ├── research/      # EDA and research notebooks
│   └── evaluation/    # Model evaluation notebooks
├── src/
│   ├── features/      # Feature engineering code
│   ├── models/        # Model definitions and training
│   └── evaluation/    # Backtesting and evaluation utilities
├── tests/             # Unit and integration tests
├── configs/           # Model and scraping configs
└── outputs/
    ├── models/        # Saved trained models
    └── predictions/   # Generated predictions
```

## Tech Stack

- **Language**: Python 3.11+
- **Data**: pandas, polars, numpy
- **Scraping**: requests, beautifulsoup4, selenium (if needed for JS-rendered pages)
- **ML**: scikit-learn, xgboost, lightgbm, pytorch (if deep learning needed)
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: Jupyter
- **Testing**: pytest

## Development Guidelines

- Always use temporal splits for train/test — never leak future data into training
- Pin dependencies in `requirements.txt`
- Keep scrapers modular: one scraper per data source
- Store intermediate data as Parquet for efficiency
- All model experiments should be reproducible (set random seeds, log parameters)
- Use `configs/` for hyperparameters and scraping settings rather than hardcoding
- Write docstrings for public functions in `src/`
- Notebooks are for exploration; production logic goes in `src/`

## Key Metrics

- **Log-loss**: primary metric for probability calibration
- **Brier score**: measures accuracy of probabilistic predictions
- **ROI**: return on investment when simulating bets against actual odds
- **Calibration**: predicted probabilities should match observed frequencies

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run scrapers
python -m scrapers.<source_name>

# Run tests
pytest tests/

# Train model
python -m src.models.train --config configs/model_config.yaml
```
