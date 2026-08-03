import json
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIGURAZIONE TICKER ---
TICKER_ETP = "IWVL.MI"  # iShares World Value Factor
TICKER_FUT = "NQ=F"     # Nasdaq Futures (Benchmark/Supporto)

# --- FUNZIONI DI CALCOLO INDICATORI ---

def calc_kama(prices, n=10, pow1=2, pow2=30):
    """Calcola la Kaufman's Adaptive Moving Average (KAMA)"""
    change = abs(prices - prices.shift(n))
    volatility = abs(prices - prices.shift(1)).rolling(n).sum()
    er = np.where(volatility == 0, 0, change / volatility)
    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2
    
    kama = np.zeros_like(prices)
    kama[:n] = prices[:n]
    for i in range(n, len(prices)):
        kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
    return pd.Series(kama, index=prices.index)

def calc_rsi(prices, period=14):
    """Calcola l'RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ao(df):
    """Calcola l'Awesome Oscillator (AO)"""
    median_price = (df['High'] + df['Low']) / 2
    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    return ao

def fmt_list(series, dec=2):
    """Converte serie pandas/numpy in liste Python pulite per il JSON"""
    res = []
    for v in series:
        if pd.isna(v) or math.isnan(v) or math.isinf(v):
            res.append(None)
        else:
            res.append(round(float(v), dec))
    return res

def clean_dict(obj):
    """Pulisce ricorsivamente NaN e Inf prima della serializzazione JSON"""
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
    print(f"🔄 Scaricamento dati per {TICKER_ETP}...")
    
    # 1. Download Dati
    etp_df = yf.Ticker(TICKER_ETP).history(period="max")
    
    if etp_df.empty:
        raise ValueError(f"Impossibile recuperare i dati per {TICKER_ETP}")

    # Filtriamo solo gli ultimi 756 giorni lavorativi (circa 3 anni)
    etp = etp_df.tail(756).copy()
    
    # 2. Calcolo Indicatori
    closes = etp['Close']
    highs = etp['High']
    lows = etp['Low']
    
    kama_fast = calc_kama(closes, n=10)
    kama_slow = calc_kama(closes, n=30)
    rsi14 = calc_rsi(closes, period=14)
    rsi5 = calc_rsi(closes, period=5)
    ao = calc_ao(etp)
    
    # Calcolo Efficiency Ratio (ER)
    change = abs(closes - closes.shift(10))
    volatility = abs(closes - closes.shift(1)).rolling(10).sum()
    er_series = np.where(volatility == 0, 0, change / volatility)
    er_val = round(float(er_series[-1]), 4) if len(er_series) > 0 else 0.0

    # Segnali RAPTOR Base (1 se KAMA Fast > KAMA Slow, altrimenti 0)
    signals = np.where(kama_fast > kama_slow, 1, 0).tolist()

    # Antonacci Dual Momentum 12M (Rendimento a 252 giorni)
    ret_12m_series = closes.pct_change(252) * 100
    ret_12m_val = round(float(ret_12m_series.iloc[-1]), 2) if not pd.isna(ret_12m_series.iloc[-1]) else 0.0
    ant_signal = "LONG" if ret_12m_val > 0 else "CASH"

    # 3. Preparazione Output JSON
    now = datetime.now()
    execution_type = "close" if now.hour >= 16 else "morning"

    output = {
        "execution_type": execution_type,
        "updated_at": now.isoformat(),
        "updated_display": now.strftime("%d/%m/%Y %H:%M"),
        "etp": {
            "dates": [d.strftime("%Y-%m-%d") for d in etp.index],
            "closes": fmt_list(closes, 2),
            "highs": fmt_list(highs, 2),
            "lows": fmt_list(lows, 2),
            "volumes": [int(v) if not pd.isna(v) else 0 for v in etp['Volume']],
            "kama_fast": fmt_list(kama_fast, 2),
            "kama_slow": fmt_list(kama_slow, 2),
            "rsi14": fmt_list(rsi14, 2),
            "rsi5": fmt_list(rsi5, 2),
            "ao": fmt_list(ao, 2),
            "er": er_val,
            "signals": signals
        },
        "antonacci_etp_latest": {
            "ret_12m": ret_12m_val,
            "signal": ant_signal
        }
    }

    # Pulizia di sicurezza da valori nulli/invalidi
    output = clean_dict(output)

    # 4. Salvataggio
    with open('iwvl.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    print("✅ iwvl.json generato e popolato con successo!")

if __name__ == "__main__":
    main()
