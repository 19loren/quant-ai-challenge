START_DATE = "2015-01-01"
END_DATE = None

TOP_N = 20

LOOKBACK_MOM = 252
SKIP_MOM = 21

VOL_LOOKBACK = 60

MIN_HISTORY_MONTHS = 24
MIN_VOLUME = 5_000_000

REBALANCE = "M"

REGIME_MA = 200
REGIME_BAND = 0.05

RISK_ON = {
    "momentum":0.50,
    "quality":0.35,
    "risk":0.15
}

RISK_OFF = {
    "momentum":0.25,
    "quality":0.55,
    "risk":0.20
}

ONE_WAY_COST = 0.00125
