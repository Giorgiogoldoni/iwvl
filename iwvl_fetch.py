#!/usr/bin/env python3
"""
iwvl_fetch.py
Scarica lo storico di IWVL.MI, applica il motore scannerv3 (engine_scannerv3.py, copia
IDENTICA di fetch_azionario.py del repo azionario) e scrive:
  - iwvl.json                     riepilogo (stesso schema di azionario.json, un solo titolo)
  - data/charts/IWVL_MI.json      serie storiche + indicatori per il grafico
  - regole/IWVL_MI_Regole.html    scheda regole operative (stampabile in PDF)

Il motore NON va modificato qui: per aggiornare la logica dei segnali si ricopia
fetch_azionario.py sopra engine_scannerv3.py.
"""

import json
import math
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import engine_scannerv3 as eng

# --- CONFIGURAZIONE ---
TICKER = "IWVL.MI"
NOME = "iShares Edge MSCI World Value Factor UCITS ETF"
PERIODO = "2y"                          # 2 anni di segnali ed entrate/uscite (standard scannerv3)
TZ = ZoneInfo("Europe/Rome")
CHIUSURA_BORSA = time(17, 45)           # Borsa Italiana chiude 17:30 (+ margine consolidamento)
MAX_ETA_ULTIMA_BARRA_GG = 6             # oltre: il job fallisce invece di pubblicare dati vecchi

ROOT = Path(__file__).parent
OUT_JSON = ROOT / "iwvl.json"
FILE_STEM = TICKER.replace(".", "_").replace("-", "_")      # IWVL_MI (come fileFor() nel browser)


def pulisci(obj):
    """Converte tipi numpy e sostituisce NaN/Inf con None (JSON valido per il browser)."""
    if isinstance(obj, dict):
        return {k: pulisci(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pulisci(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return obj


def scarica(now):
    df = yf.Ticker(TICKER).history(period=PERIODO, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Impossibile recuperare i dati per {TICKER}")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Esclude la barra del giorno se la seduta e' ancora in corso (dato parziale)
    parziale = False
    seduta_in_corso = now.weekday() < 5 and now.time() < CHIUSURA_BORSA
    if len(df) and seduta_in_corso and df.index[-1].date() == now.date():
        df = df.iloc[:-1]
        parziale = True
    if df.empty:
        raise ValueError("Nessuna barra completa disponibile")

    # Controllo staleness: meglio far fallire il job che pubblicare dati vecchi
    ultima = df.index[-1].date()
    eta = (now.date() - ultima).days
    if eta > MAX_ETA_ULTIMA_BARRA_GG:
        raise ValueError(f"Ultima barra del {ultima} ({eta} giorni fa): dati troppo vecchi, file non aggiornati")
    return df, ultima, parziale


def main():
    now = datetime.now(TZ)
    print(f"Scaricamento {TICKER} ({now.strftime('%d/%m/%Y %H:%M')} Europe/Rome)...")
    df, ultima, parziale = scarica(now)

    out = eng.compute_indicators(df)
    if out is None:
        raise ValueError("Storico insufficiente per calcolare gli indicatori")
    summary, chart = out

    tv = eng.tv_symbol(TICKER)                                  # MIL:IWVL
    riga = {
        "regione": "ISHARES", "nome": NOME, "ticker": TICKER, "settore": "ETF Value World",
        "paese": "World", "exchange": None, "tv_symbol": tv, **summary,
    }

    (ROOT / "data" / "charts").mkdir(parents=True, exist_ok=True)
    (ROOT / "regole").mkdir(parents=True, exist_ok=True)

    (ROOT / "data" / "charts" / f"{FILE_STEM}.json").write_text(
        json.dumps(pulisci(chart), ensure_ascii=False, allow_nan=False), encoding="utf-8")

    (ROOT / "regole" / f"{FILE_STEM}_Regole.html").write_text(
        eng.build_regole_html(NOME, TICKER, summary, {"regione": "ISHARES", "tv_symbol": tv}),
        encoding="utf-8")

    OUT_JSON.write_text(json.dumps(pulisci({
        "generato": now.isoformat(timespec="seconds"),
        "generato_display": now.strftime("%d/%m/%Y %H:%M"),
        "timezone": "Europe/Rome",
        "ultima_barra": ultima.strftime("%Y-%m-%d"),
        "ultima_barra_display": ultima.strftime("%d/%m/%Y"),
        "barra_parziale_esclusa": parziale,
        "totale": 1,
        "errori": [],
        "titoli": [riga],
    }), ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print(f"OK: ultima barra {ultima}, barra parziale esclusa: {parziale}, "
          f"segnale {summary['segnale']}, rating {summary['rating']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                      # job rosso su GitHub Actions
        print(f"ERRORE: {e}", file=sys.stderr)
        raise
