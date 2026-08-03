import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIGURAZIONE TICKER ---
TICKER_ETP = "IWVL.MI"  # Ticker per iShares World Value Factor
TICKER_FUT = "NQ=F"      # Futures Nasdaq (o quello usato come benchmark di supporto)

def main():
    print(f"Scaricamento dati per {TICKER_ETP} e {TICKER_FUT}...")
    
    # 1. Download Dati da Yahoo Finance
    etp = yf.Ticker(TICKER_ETP).history(period="max")
    fut = yf.Ticker(TICKER_FUT).history(period="max")

    if etp.empty:
        raise ValueError(f"Impossibile recuperare i dati per {TICKER_ETP}")

    # (Qui inserisci la stessa logica di calcolo KAMA, RSI, Dual Momentum, SAR, ADX già usata in iwmo_fetch.py)
    # ...

    # 2. Generazione Output JSON
    now = datetime.now()
    execution_type = "close" if now.hour >= 16 else "morning"

    output = {
        "execution_type": execution_type,
        "updated_at": now.isoformat(),
        "updated_display": now.strftime("%d/%m/%Y %H:%M"),
        # Struttura dei dati per il frontend HTML
        "etp": {
            # "dates": dates_list,
            # "closes": closes_list,
            # "kama_fast": kama_list,
            # ...
        }
    }

    # 3. Salvataggio su iwvl.json
    with open('iwvl.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    print(f"✅ iwvl.json generato con successo!")

if __name__ == "__main__":
    main()
