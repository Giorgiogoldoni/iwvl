import json
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# --- CONFIGURAZIONE ---
TICKER_ETP = "IWVL.MI"                 # iShares Edge MSCI World Value Factor
TZ = ZoneInfo("Europe/Rome")
CHIUSURA_BORSA = time(17, 45)          # Borsa Italiana chiude 17:30 (+ margine consolidamento dati)
BARRE_MAX = 756                        # ~3 anni di sedute
MAX_ETA_ULTIMA_BARRA_GG = 6            # oltre questa soglia il job fallisce (dati vecchi)
OUTPUT_FILE = "iwvl.json"


# --- FUNZIONI DI CALCOLO INDICATORI ---
def calc_er(prices, n=10):
    """Efficiency Ratio di Kaufman."""
    change = (prices - prices.shift(n)).abs()
    volatility = prices.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.fillna(0.0).where(volatility.notna(), np.nan)


def calc_kama(prices, n=10, pow1=2, pow2=30):
    """Kaufman Adaptive Moving Average. Lavora su array numpy (nessun accesso per indice-data)."""
    p = prices.to_numpy(dtype=float)
    kama = np.full(len(p), np.nan)
    if len(p) <= n:
        return pd.Series(kama, index=prices.index)

    er = calc_er(prices, n).fillna(0.0).to_numpy()
    fast_sc = 2 / (pow1 + 1)
    slow_sc = 2 / (pow2 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama[:n] = p[:n]
    for i in range(n, len(p)):
        kama[i] = kama[i - 1] + sc[i] * (p[i] - kama[i - 1])
    return pd.Series(kama, index=prices.index)


def calc_rsi(prices, period=14):
    """RSI con smoothing di Wilder (coerente con TradingView)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0).where(avg_loss.notna(), np.nan)


def calc_ao(df):
    """Awesome Oscillator."""
    median_price = (df["High"] + df["Low"]) / 2
    return median_price.rolling(5).mean() - median_price.rolling(34).mean()


def fmt_list(series, dec=2):
    """Serie pandas/numpy -> lista Python pulita per il JSON."""
    res = []
    for v in series:
        if pd.isna(v) or math.isnan(v) or math.isinf(v):
            res.append(None)
        else:
            res.append(round(float(v), dec))
    return res


def clean_dict(obj):
    """Pulisce ricorsivamente NaN e Inf prima della serializzazione JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    elif isinstance(obj, dict):
        return {k: clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict(v) for v in obj]
    return obj


# --- MAIN ---
def main():
    now = datetime.now(TZ)
    print(f"Scaricamento dati per {TICKER_ETP} ({now.strftime('%d/%m/%Y %H:%M')} Europe/Rome)...")

    etp_df = yf.Ticker(TICKER_ETP).history(period="max", auto_adjust=True)
    if etp_df.empty:
        raise ValueError(f"Impossibile recuperare i dati per {TICKER_ETP}")

    etp_df = etp_df.dropna(subset=["Close", "High", "Low"])

    # Esclude la barra del giorno se la seduta e' ancora in corso (dato parziale)
    partial_excluded = False
    seduta_in_corso = now.weekday() < 5 and now.time() < CHIUSURA_BORSA
    if len(etp_df) > 0 and seduta_in_corso and etp_df.index[-1].date() == now.date():
        etp_df = etp_df.iloc[:-1]
        partial_excluded = True

    if etp_df.empty:
        raise ValueError("Nessuna barra completa disponibile")

    # Controllo staleness: meglio far fallire il job che pubblicare dati vecchi
    last_date = etp_df.index[-1].date()
    eta_gg = (now.date() - last_date).days
    if eta_gg > MAX_ETA_ULTIMA_BARRA_GG:
        raise ValueError(
            f"Ultima barra del {last_date} ({eta_gg} giorni fa): dati troppo vecchi, JSON non aggiornato"
        )

    etp = etp_df.tail(BARRE_MAX).copy()
    closes = etp["Close"]

    # Indicatori
    kama_fast = calc_kama(closes, n=10)
    kama_slow = calc_kama(closes, n=30)
    rsi14 = calc_rsi(closes, period=14)
    rsi5 = calc_rsi(closes, period=5)
    ao = calc_ao(etp)

    er_series = calc_er(closes, n=10)
    er_last = er_series.iloc[-1]
    er_val = round(float(er_last), 4) if not pd.isna(er_last) else 0.0

    # Segnale RAPTOR base: 1 se KAMA fast > KAMA slow, altrimenti 0
    signals = np.where(kama_fast > kama_slow, 1, 0)
    changes = np.flatnonzero(np.diff(signals) != 0)
    start = int(changes[-1]) + 1 if len(changes) else 0
    signal_since = etp.index[start].strftime("%Y-%m-%d")
    signal_bars = int(len(signals) - start)

    # Momentum assoluto 12M (252 sedute): NON e' un Dual Momentum completo
    ret_12m_series = closes.pct_change(252) * 100
    ret_12m_last = ret_12m_series.iloc[-1]
    ret_12m_val = round(float(ret_12m_last), 2) if not pd.isna(ret_12m_last) else 0.0
    ant_signal = "LONG" if ret_12m_val > 0 else "CASH"

    output = {
        "execution_type": "close" if now.time() >= CHIUSURA_BORSA else "morning",
        "updated_at": now.isoformat(),
        "updated_display": now.strftime("%d/%m/%Y %H:%M"),
        "timezone": "Europe/Rome",
        "last_bar_date": last_date.strftime("%Y-%m-%d"),
        "last_bar_display": last_date.strftime("%d/%m/%Y"),
        "partial_bar_excluded": partial_excluded,
        "etp": {
            "dates": [d.strftime("%Y-%m-%d") for d in etp.index],
            "closes": fmt_list(closes, 2),
            "highs": fmt_list(etp["High"], 2),
            "lows": fmt_list(etp["Low"], 2),
            "volumes": [int(v) if not pd.isna(v) else 0 for v in etp["Volume"]],
            "kama_fast": fmt_list(kama_fast, 2),
            "kama_slow": fmt_list(kama_slow, 2),
            "rsi14": fmt_list(rsi14, 2),
            "rsi5": fmt_list(rsi5, 2),
            "ao": fmt_list(ao, 2),
            "er": er_val,
            "signals": signals.tolist(),
            "signal_since": signal_since,
            "signal_bars": signal_bars,
        },
        "antonacci_etp_latest": {
            "ret_12m": ret_12m_val,
            "signal": ant_signal,
            "tipo": "momentum assoluto 12M (rendimento > 0), non Dual Momentum",
        },
    }

    output = clean_dict(output)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print(f"{OUTPUT_FILE} generato: ultima barra {last_date}, barra parziale esclusa: {partial_excluded}")


if __name__ == "__main__":
    main()
