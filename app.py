# ==============================================================
# RV PREDICTOR — FLASK WEB APP  (Full Website Edition)
# ==============================================================
#
#  SETUP:  pip install flask pandas openpyxl
#  RUN:    python app.py
#  LOCAL:  http://localhost:5000
#  PHONE:  http://YOUR-PC-IP:5000   (run ipconfig to find IP)
#  PUBLIC: set ENABLE_NGROK = True  (pip install pyngrok first)
#
# ==============================================================

import os, glob, json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent

REPORTS_DIR  = BASE_DIR / "data"
DATASET_PATH = BASE_DIR / "data" / "historical_data.csv"
ACC_FILE     = BASE_DIR / "data" / "accuracy_tracker.csv"

ENABLE_NGROK = False   # IMPORTANT
NGROK_TOKEN  = ""   # leave "" to auto-read from ngrok config file

app  = Flask(__name__, template_folder="templates")
_df_cache     = None
_df_cache_ts  = 0
CACHE_SECONDS = 300

# Automatically skip ngrok browser warning for all visitors
@app.after_request
def skip_ngrok_warning(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ── DATASET LOADER ─────────────────────────────────────────────

def get_dataset():
    global _df_cache, _df_cache_ts
    now = datetime.now().timestamp()
    if _df_cache is None or (now - _df_cache_ts) > CACHE_SECONDS:
        try:
            df = pd.read_csv(DATASET_PATH)

            # ── AUTO-DETECT column names (handles different CSV formats) ──
            cols = {c.lower().strip(): c for c in df.columns}

            # Market/Mandi column
            for candidate in ['market','mandi','Market','Mandi','district','District']:
                if candidate.lower() in cols:
                    df = df.rename(columns={cols[candidate.lower()]: 'market'})
                    break

            # Price column
            for candidate in ['price','modal_price','Modal Price','modal price','Price','avg_price','Avg Price']:
                if candidate.lower().replace(' ','_') in {c.lower().replace(' ','_') for c in df.columns}:
                    actual = next(c for c in df.columns if c.lower().replace(' ','_') == candidate.lower().replace(' ','_'))
                    df = df.rename(columns={actual: 'price'})
                    break

            # Arrival column
            for candidate in ['arrival','arrivals','Arrival','Arrivals','total_arrival','quantity']:
                if candidate.lower() in cols:
                    df = df.rename(columns={cols[candidate.lower()]: 'arrival'})
                    break

            # Date column
            for candidate in ['date','Date','trade_date','arrival_date']:
                if candidate.lower() in cols:
                    df = df.rename(columns={cols[candidate.lower()]: 'date'})
                    break

            # Clean price column
            if 'price' in df.columns:
                if df['price'].dtype == object:
                    df['price'] = pd.to_numeric(
                        df['price'].astype(str).str.replace(',','').str.strip(),
                        errors='coerce'
                    )

            if 'arrival' not in df.columns:
                df['arrival'] = 0

            df['date'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

            for col in ['min_price', 'max_price']:
                if col in df.columns and df[col].dtype == object:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', ''), errors='coerce'
                    )

            print(f"[Dataset] Loaded {len(df)} rows. Columns: {list(df.columns[:8])}")
            if 'market' in df.columns:
                print(f"[Dataset] Mandis found: {sorted(df['market'].unique().tolist())[:5]}...")

            _df_cache    = df
            _df_cache_ts = now
        except Exception as e:
            print(f'[Dataset ERROR] {e}')
            import traceback; traceback.print_exc()
            return pd.DataFrame()
    return _df_cache


def get_signals():
    patterns = [
        os.path.join(REPORTS_DIR, "Daily_Signals_*.xlsx"),
        os.path.join(REPORTS_DIR, "Daily_Signals_*.csv"),
        os.path.join(REPORTS_DIR, "RV_Signals_*.xlsx"),
        os.path.join(REPORTS_DIR, "AI_Prediction_*.xlsx"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None, None
    latest = max(files, key=os.path.getmtime)
    ext    = Path(latest).suffix.lower()
    df     = pd.read_csv(latest) if ext == '.csv' else pd.read_excel(latest, sheet_name=0)
    rdate  = datetime.fromtimestamp(os.path.getmtime(latest)).strftime('%d %b %Y %H:%M')
    return df, rdate


def clean_signal_row(row):
    def safe(v, default=0):
        if isinstance(v, str): return v
        try:
            if pd.isna(v): return default
            return round(float(v), 2)
        except: return default

    return {
        'mandi':    str(row.get('Mandi',    row.get('mandi', ''))),
        'price':    safe(row.get('Current Price', 0)),
        'pred1':    safe(row.get('Pred 1d ₹',  row.get('Pred 1d Rs', 0))),
        'pred3':    safe(row.get('Pred 3d ₹',  row.get('Pred 3d Rs', 0))),
        'pred7':    safe(row.get('Pred 7d ₹',  row.get('Pred 7d Rs', 0))),
        'pred14':   safe(row.get('Pred 14d ₹', row.get('Pred 14d Rs', 0))),
        'ret1':     safe(row.get('Ret 1d %',  0)),
        'ret3':     safe(row.get('Ret 3d %',  0)),
        'ret7':     safe(row.get('Ret 7d %',  0)),
        'ret14':    safe(row.get('Ret 14d %', 0)),
        'conf1':    safe(row.get('Conf 1d %',  0)),
        'conf3':    safe(row.get('Conf 3d %',  0)),
        'conf7':    safe(row.get('Conf 7d %',  0)),
        'conf14':   safe(row.get('Conf 14d %', 0)),
        'crash':    safe(row.get('Crash Risk %', 0)),
        'arrival':  safe(row.get('Arrival Pressure', 1)),
        'zone':     str(row.get('Price Zone', '')),
        'regime':   str(row.get('Regime', row.get('Market Regime', ''))),
        'trend':    str(row.get('Trend Shape', '')),
        'entry':    str(row.get('Entry Signal', '')),
        'plan':     str(row.get('Planning Signal', '')),
        'score':    safe(row.get('Procurement Score', 0)),
        'accuracy': str(row.get('Live Accuracy', 'building...')),
        'rows':     safe(row.get('Rows', 0)),
    }


# ── PAGES ──────────────────────────────────────────────────────

@app.route('/')
def home():       return render_template('index.html')

@app.route('/mandi')
def mandi():      return render_template('mandi.html')

@app.route('/compare')
def compare():    return render_template('compare.html')

@app.route('/history')
def history():    return render_template('history.html')


# ── DEBUG: visit /api/debug to see what's loaded ───────────────

@app.route('/api/debug')
def api_debug():
    df = get_dataset()
    sig_df, rdate = get_signals()

    dataset_info = {}
    if not df.empty:
        dataset_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'mandis': sorted(df['market'].unique().tolist()) if 'market' in df.columns else [],
            'date_range': [str(df['date'].min().date()), str(df['date'].max().date())] if 'date' in df.columns else []
        }
    else:
        dataset_info = {'error': f'Dataset empty or failed to load. Path: {DATASET_PATH}'}

    signal_info = {}
    if sig_df is not None:
        signal_info = {
            'rows': len(sig_df),
            'columns': list(sig_df.columns),
            'report_date': rdate,
            'mandis': sig_df.get('Mandi', sig_df.get('mandi', pd.Series([]))).tolist()[:5]
        }
    else:
        signal_info = {'error': f'No signal file found in: {REPORTS_DIR}'}

    return jsonify({
        'dataset': dataset_info,
        'signals': signal_info,
        'template_folder': app.template_folder,
        'paths': {
            'REPORTS_DIR': str(REPORTS_DIR),
            'DATASET_PATH': str(DATASET_PATH),
         }
    })


# ── API: SIGNALS ───────────────────────────────────────────────

@app.route('/api/signals')
def api_signals():
    df, rdate = get_signals()
    if df is None:
        return jsonify({'error': 'No signal file found', 'data': []})
    score_col = 'Procurement Score'
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)
    data = [clean_signal_row(r) for _, r in df.iterrows()
            if str(r.get('Mandi', r.get('mandi', ''))).strip()]
    return jsonify({'data': data, 'report_date': rdate, 'total': len(data)})


# ── API: MANDI PRICE HISTORY ───────────────────────────────────

@app.route('/api/history/<mandi>')
def api_history(mandi):
    df  = get_dataset()
    if df.empty:
        return jsonify({'error': f'Dataset not available. Check path: {DATASET_PATH}', 'dates':[], 'prices':[], 'arrivals':[], 'seasons':[]})

    if 'market' not in df.columns:
        return jsonify({'error': f'No market column found. Available: {list(df.columns)}', 'dates':[], 'prices':[], 'arrivals':[], 'seasons':[]})

    days = int(request.args.get('days', 9999))
    mdf  = df[df['market'] == mandi].sort_values('date')
    if days < 9999:
        mdf = mdf.tail(days)

    if mdf.empty:
        return jsonify({'error': f'No data for mandi: {mandi}', 'dates':[], 'prices':[], 'arrivals':[], 'seasons':[]})

    # Season flags
    seasons = []
    for _, row in mdf.iterrows():
        if row.get('harvest_season', 0) == 1:   seasons.append('harvest')
        elif row.get('festival_season', 0) == 1: seasons.append('festival')
        elif row.get('off_season', 0) == 1:      seasons.append('off')
        else:                                     seasons.append('normal')

    arr_col = mdf['arrival'] if 'arrival' in mdf.columns else pd.Series([0]*len(mdf))

    return jsonify({
        'dates':    mdf['date'].dt.strftime('%Y-%m-%d').tolist(),
        'prices':   mdf['price'].round(0).tolist(),
        'min_price':mdf['min_price'].round(0).tolist() if 'min_price' in mdf.columns else [],
        'max_price':mdf['max_price'].round(0).tolist() if 'max_price' in mdf.columns else [],
        'arrivals': arr_col.fillna(0).round(1).tolist(),
        'seasons':  seasons,
        'mandi':    mandi,
    })


# ── API: ALL MANDIS LIST ───────────────────────────────────────

@app.route('/api/mandis')
def api_mandis():
    # PRIMARY: try to get from dataset CSV
    df = get_dataset()
    if not df.empty and 'market' in df.columns:
        mandis = sorted(df['market'].dropna().astype(str).unique().tolist())
        mandis = [m for m in mandis if m and m != 'nan']
        if mandis:
            return jsonify(mandis)

    # FALLBACK: get from signals file (always available)
    print('[api/mandis] Dataset empty/missing market col — falling back to signals file')
    sig_df, _ = get_signals()
    if sig_df is not None:
        col = 'Mandi' if 'Mandi' in sig_df.columns else 'mandi' if 'mandi' in sig_df.columns else None
        if col:
            mandis = sorted(sig_df[col].dropna().astype(str).unique().tolist())
            mandis = [m for m in mandis if m and m != 'nan']
            print(f'[api/mandis] Got {len(mandis)} mandis from signals file')
            return jsonify(mandis)

    print('[api/mandis] No mandis found anywhere!')
    return jsonify([])


# ── API: COMPARE MANDIS ────────────────────────────────────────

@app.route('/api/compare')
def api_compare():
    selected = request.args.get('mandis', '').split(',')
    selected = [m.strip() for m in selected if m.strip()]
    days     = int(request.args.get('days', 365))

    df = get_dataset()
    if df.empty or not selected:
        return jsonify({'error': 'No data'})

    result = {}
    for mandi in selected:
        mdf = df[df['market'] == mandi].sort_values('date').tail(days)
        if mdf.empty: continue
        result[mandi] = {
            'dates':    mdf['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices':   mdf['price'].round(0).tolist(),
            'arrivals': mdf['arrival'].round(1).tolist(),
            'latest':   float(mdf['price'].iloc[-1]),
            'avg':      float(mdf['price'].mean().round(0)),
            'min':      float(mdf['price'].min()),
            'max':      float(mdf['price'].max()),
        }

    # Spread analysis — cheapest vs most expensive today
    latest_prices = {m: v['latest'] for m, v in result.items()}
    cheapest  = min(latest_prices, key=latest_prices.get)
    costliest = max(latest_prices, key=latest_prices.get)
    spread    = round(latest_prices[costliest] - latest_prices[cheapest], 0)

    return jsonify({
        'mandis': result,
        'spread': {
            'cheapest':  cheapest,
            'costliest': costliest,
            'diff':      spread,
        }
    })


# ── API: SEASONALITY ───────────────────────────────────────────

@app.route('/api/seasonality')
def api_seasonality():
    df = get_dataset()
    if df.empty:
        return jsonify({'error': 'No data'})

    mandi = request.args.get('mandi', None)
    if mandi:
        df = df[df['market'] == mandi]

    # Monthly avg price + arrival
    df['month'] = df['date'].dt.month
    df['week']  = df['date'].dt.isocalendar().week.astype(int)
    df['year']  = df['date'].dt.year

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    monthly = df.groupby('month').agg(
        avg_price   = ('price',   'mean'),
        avg_arrival = ('arrival', 'mean'),
        min_price   = ('price',   'min'),
        max_price   = ('price',   'max'),
    ).round(0).reset_index()

    weekly = df.groupby('week').agg(
        avg_price   = ('price',   'mean'),
        avg_arrival = ('arrival', 'mean'),
    ).round(0).reset_index()

    yearly = df.groupby('year').agg(
        avg_price   = ('price',   'mean'),
        total_arr   = ('arrival', 'sum'),
    ).round(0).reset_index()

    return jsonify({
        'monthly': {
            'months':    monthly['month'].tolist(),
            'labels':    [month_names[m-1] for m in monthly['month'].tolist()],
            'prices':    monthly['avg_price'].tolist(),
            'arrivals':  monthly['avg_arrival'].tolist(),
            'min':       monthly['min_price'].tolist(),
            'max':       monthly['max_price'].tolist(),
        },
        'weekly': {
            'weeks':     weekly['week'].tolist(),
            'prices':    weekly['avg_price'].tolist(),
            'arrivals':  weekly['avg_arrival'].tolist(),
        },
        'yearly': {
            'years':     yearly['year'].tolist(),
            'prices':    yearly['avg_price'].tolist(),
            'arrivals':  yearly['total_arr'].tolist(),
        }
    })


# ── API: CRASH HEATMAP ─────────────────────────────────────────

@app.route('/api/heatmap')
def api_heatmap():
    sdf, _ = get_signals()
    if sdf is None:
        return jsonify([])

    result = []
    for _, row in sdf.iterrows():
        mandi = str(row.get('Mandi', row.get('mandi', ''))).strip()
        if not mandi: continue
        crash  = float(row.get('Crash Risk %', 0) or 0)
        score  = float(row.get('Procurement Score', 0) or 0)
        price  = float(row.get('Current Price', 0) or 0)
        entry  = str(row.get('Entry Signal', ''))
        result.append({
            'mandi': mandi, 'crash': crash,
            'score': score, 'price': price, 'entry': entry,
        })

    result.sort(key=lambda x: x['crash'], reverse=True)
    return jsonify(result)


# ── API: PRICE SPREAD ──────────────────────────────────────────

@app.route('/api/spread')
def api_spread():
    sdf, rdate = get_signals()
    if sdf is None:
        return jsonify({'error': 'No data'})

    rows = []
    for _, row in sdf.iterrows():
        mandi = str(row.get('Mandi', row.get('mandi', ''))).strip()
        price = float(row.get('Current Price', 0) or 0)
        score = float(row.get('Procurement Score', 0) or 0)
        entry = str(row.get('Entry Signal', ''))
        crash = float(row.get('Crash Risk %', 0) or 0)
        if mandi and price > 0:
            rows.append({'mandi': mandi, 'price': price,
                         'score': score, 'entry': entry, 'crash': crash})

    if not rows: return jsonify({'error': 'No data'})

    rows.sort(key=lambda x: x['price'])
    cheapest = rows[0]
    avg      = round(sum(r['price'] for r in rows) / len(rows), 0)

    return jsonify({
        'mandis':   rows,
        'cheapest': cheapest,
        'avg':      avg,
        'spread':   round(rows[-1]['price'] - rows[0]['price'], 0),
        'date':     rdate,
    })


# ── MAIN ───────────────────────────────────────────────────────

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def try_copy_to_clipboard(text):
    """Silently try to copy URL to clipboard on Windows."""
    try:
        import subprocess
        subprocess.run(['clip'], input=text.encode('utf-8'), check=True)
        return True
    except Exception:
        return False

def read_ngrok_token():
    """Read ngrok authtoken from config file on disk."""
    import re as _r
    win_cfg = os.path.join(os.path.expanduser("~"), "AppData", "Local", "ngrok", "ngrok.yml")
    paths = [win_cfg, os.path.expanduser("~/.config/ngrok/ngrok.yml")]
    for p in paths:
        try:
            if os.path.exists(p):
                txt = open(p).read()
                m = _r.search("authtoken" + r"[:\s]+" + r"([\w-]+)", txt)
                if m:
                    return m.group(1)
        except Exception:
            pass
    return None


def start_ngrok_tunnel(port):
    """Start ngrok only if a real authtoken is found. Returns (url, error_str)."""
    token = NGROK_TOKEN if NGROK_TOKEN else read_ngrok_token()
    if not token:
        return None, "no_token"

    # Try modern ngrok Python SDK (pip install ngrok)
    try:
        import ngrok as _ng
        os.environ["NGROK_AUTHTOKEN"] = token
        listener = _ng.forward(port)
        url = listener.url()
        if url:
            return url, None
    except Exception:
        pass

    # Try ngrok.exe binary in PATH
    try:
        import subprocess, time, urllib.request, json as _json
        subprocess.run(["taskkill", "/f", "/im", "ngrok.exe"], capture_output=True)
        time.sleep(0.5)
        subprocess.Popen(
            ["ngrok", "http", str(port), "--authtoken=" + token],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        resp = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=4)
        tunnels = _json.loads(resp.read()).get("tunnels", [])
        for t in tunnels:
            if t.get("proto") == "https":
                return t["public_url"], None
        if tunnels:
            return tunnels[0]["public_url"], None
    except Exception:
        pass

    return None, "failed"



# ── MAIN ───────────────────────────────────────────────────────

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def try_copy_to_clipboard(text):
    try:
        import subprocess
        subprocess.run(['clip'], input=text.encode('utf-8'), check=True)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    app.run()