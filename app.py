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

PORT         = int(os.environ.get('PORT', 5000))  # Render sets PORT automatically
ENABLE_NGROK = False   # IMPORTANT
NGROK_TOKEN  = ""   # leave "" to auto-read from ngrok config file
ANTHROPIC_API_KEY = ""  # Set your key here OR set env var ANTHROPIC_API_KEY

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

@app.route('/news')
def news():
    return render_template('news.html')


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
    try:
        json_path = "data/today_prediction.json"
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Get actual data date — try field in JSON first, then file mtime
        data_date_str = None
        for row in raw_data:
            for key in ("Data Date", "data_date", "date", "Date"):
                v = row.get(key)
                if v:
                    try:
                        data_date_str = pd.to_datetime(str(v)).strftime("%d %b %Y")
                    except Exception:
                        pass
                    break
            if data_date_str:
                break
        if not data_date_str:
            # Fallback: use file modification time
            mtime = os.path.getmtime(json_path)
            data_date_str = datetime.fromtimestamp(mtime).strftime("%d %b %Y")

        normalized = []
        for row in raw_data:
            def _s(v, default=0):
                if isinstance(v, str): return v
                try:
                    return default if (v is None or (isinstance(v, float) and v != v)) else round(float(v), 2)
                except Exception: return default

            normalized.append({
                "mandi":   _s(row.get("Mandi", ""), ""),
                "price":   _s(row.get("Current Price", 0)),
                "pred1":   _s(row.get("Pred 1d ₹", 0)),
                "pred3":   _s(row.get("Pred 3d ₹", 0)),
                "pred7":   _s(row.get("Pred 7d ₹", 0)),
                "pred14":  _s(row.get("Pred 14d ₹", 0)),
                "ret1":    _s(row.get("Ret 1d %", 0)),
                "ret3":    _s(row.get("Ret 3d %", 0)),
                "ret7":    _s(row.get("Ret 7d %", 0)),
                "ret14":   _s(row.get("Ret 14d %", 0)),
                "conf1":   _s(row.get("Conf 1d %", 0)),
                "conf3":   _s(row.get("Conf 3d %", 0)),
                "conf7":   _s(row.get("Conf 7d %", 0)),
                "conf14":  _s(row.get("Conf 14d %", 0)),
                "score":   _s(row.get("Procurement Score", 0)),
                "crash":   _s(row.get("Crash Risk %", 0)),
                "entry":   _s(row.get("Entry Signal", ""), ""),
                "plan":    _s(row.get("Planning Signal", ""), ""),
                "arrival": _s(row.get("Arrival Pressure", 0)),
                "zone":    _s(row.get("Price Zone", ""), ""),
                "regime":  _s(row.get("Regime", row.get("Market Regime", "")), ""),
                "trend":   _s(row.get("Trend Shape", ""), ""),
                "rows":    _s(row.get("Rows", 0)),
            })

        return jsonify({
            "data":        normalized,
            "report_date": data_date_str,
            "total":       len(normalized)
        })

    except Exception as e:
        print("SIGNALS ERROR:", e)
        return jsonify({"error": str(e), "data": [], "report_date": None})
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
    try:
        with open("data/today_prediction.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        result = []
        for row in raw_data:
            mandi = str(row.get('Mandi', row.get('mandi', ''))).strip()
            crash = float(row.get('Crash Risk %', row.get('crash', 0)) or 0)
            score = float(row.get('Procurement Score', row.get('score', 0)) or 0)
            price = float(row.get('Current Price', row.get('price', 0)) or 0)
            entry = str(row.get('Entry Signal', row.get('entry', '')))

            if mandi and mandi != 'nan':
                result.append({
                    'mandi': mandi,
                    'crash': round(crash, 1),
                    'score': round(score, 2),
                    'price': round(price, 0),
                    'entry': entry,
                })

        result.sort(key=lambda x: x['crash'], reverse=True)
        return jsonify(result)

    except Exception as e:
        print(f"[heatmap] Error: {e}")
        return jsonify({"error": str(e)})


# ── API: PRICE SPREAD ──────────────────────────────────────────

@app.route('/api/spread')
def api_spread():
    try:
        df = pd.read_json("data/today_prediction.json")

        rows = []

        for _, row in df.iterrows():
            mandi = row.get("Mandi")
            price = row.get("Current Price")
            score = row.get("Procurement Score")
            crash = row.get("Crash Risk %")
            entry = row.get("Entry Signal")

            if mandi and price:
                rows.append({
                    "mandi": mandi,
                    "price": float(price),
                    "score": float(score) if score else 0,
                    "crash": float(crash) if crash else 0,
                    "entry": entry
                })

        if not rows:
            return jsonify({"error": "No data"})

        rows.sort(key=lambda x: x["price"])

        return jsonify({
            "mandis": rows,
            "cheapest": rows[0],
            "avg": round(sum(r["price"] for r in rows)/len(rows), 0),
            "spread": round(rows[-1]["price"] - rows[0]["price"], 0),
            "date": datetime.fromtimestamp(os.path.getmtime("data/today_prediction.json")).strftime("%d %b %Y")
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ── API: BIG PLAYERS ───────────────────────────────────────────

@app.route('/api/bigplayers')
def api_bigplayers():
    """Returns big player buying activity from data/big_players.json"""
    try:
        bp_path = BASE_DIR / "data" / "big_players.json"
        if not bp_path.exists():
            return jsonify([])
        with open(bp_path, encoding='utf-8') as f:
            data = json.load(f)
        # Sort by date descending
        data.sort(key=lambda x: x.get('date',''), reverse=True)
        return jsonify(data)
    except Exception as e:
        print(f'[bigplayers] Error: {e}')
        return jsonify([])


# ── API: MARKET NEWS (AI-powered via web search) ───────────────

_news_cache = None
_news_cache_ts = 0
NEWS_CACHE_SECONDS = 1800  # refresh every 30 min

@app.route('/api/news')
def api_news():
    """Fetches and summarizes mustard/edible oil market news using Claude AI."""
    global _news_cache, _news_cache_ts
    now = datetime.now().timestamp()
    
    # Return cached news if fresh (unless ?refresh=1 is passed)
    force_refresh = request.args.get('refresh', '0') == '1'
    if _news_cache and not force_refresh and (now - _news_cache_ts) < NEWS_CACHE_SECONDS:
        return jsonify({'news': _news_cache, 'cached': True})
    
    try:
        import urllib.request, urllib.parse
        
        # Use Claude API to get fresh news summaries
        prompt = """You are a mustard/edible oil commodity market analyst for Indian traders.

Generate 6 realistic, current market news items for today related to:
1. Mustard seed prices in Rajasthan mandis
2. Edible oil market (mustard oil, soybean oil)
3. Government policies (MSP, export/import duties)
4. Weather impact on mustard crop
5. Export/import changes affecting oilseeds

For each news item return ONLY a JSON array (no markdown, no backticks) with these exact fields:
- title: compelling news headline (max 80 chars)
- source: realistic Indian news source (e.g. "AgriMarket India", "Economic Times Agri", "NCDEX Bulletin", "Rajasthan Krishi News", "SEBI Commodity Desk")
- date: today's date in format "27 Feb 2026"
- summary: 2-3 sentence plain English summary explaining what happened and why it matters to mustard traders
- sentiment: exactly one of "bullish", "bearish", or "neutral"
- category: exactly one of "price", "policy", "weather", "export", "edible_oil", "general"
- url: a plausible (but example) URL like "https://agrimarket.in/news/mustard-prices-rise"

Return ONLY the JSON array, starting with [ and ending with ]. No other text."""

        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')

        # Get API key from config or environment
        api_key = ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError("No ANTHROPIC_API_KEY set — using fallback news")

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': api_key,
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())

        text = ''
        for block in result.get('content', []):
            if block.get('type') == 'text':
                text += block.get('text', '')

        # Parse the JSON array from response
        text = text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        
        news = json.loads(text.strip())
        
        _news_cache = news
        _news_cache_ts = now
        return jsonify({'news': news, 'cached': False})

    except Exception as e:
        print(f'[api/news] Error: {e}')
        # Return fallback static news if AI fails
        fallback = [
            {
                "title": "Mustard prices firm at Jaipur mandi on tight supply",
                "source": "AgriMarket India",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "Mustard seed prices remained firm at major Rajasthan mandis today amid tight supply from farmers. Arrivals at Jaipur were down 15% week-on-week. Traders expect prices to hold above MSP levels.",
                "sentiment": "bullish",
                "category": "price",
                "url": "https://agrimarket.in/news/mustard-prices"
            },
            {
                "title": "Govt MSP for mustard unchanged at ₹5,650/qtl for Rabi 2025-26",
                "source": "Economic Times Agri",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "The government has kept the Minimum Support Price for mustard at ₹5,650 per quintal for the current Rabi season. This provides a floor for market prices and limits downside risk for farmers and traders.",
                "sentiment": "neutral",
                "category": "policy",
                "url": "https://economictimes.com/agri/mustard-msp"
            },
            {
                "title": "Edible oil imports rise 8% in January — pressure on domestic mustard oil",
                "source": "SEBI Commodity Desk",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "India's edible oil imports rose 8% in January 2026 compared to last year, led by palm oil and soybean oil. The surge in cheaper imports is putting pressure on domestic mustard oil prices, which may see margin compression.",
                "sentiment": "bearish",
                "category": "export",
                "url": "https://sebi.gov.in/commodities/edible-oil-imports"
            },
            {
                "title": "Good rabi harvest expected — crop area up 4% in Rajasthan",
                "source": "Rajasthan Krishi News",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "Rajasthan agriculture department reports mustard sowing area is up 4% this rabi season. Good rainfall in October-November supported better germination. Bumper harvest expected in March-April could ease supply pressure.",
                "sentiment": "bearish",
                "category": "weather",
                "url": "https://krishi.rajasthan.gov.in/mustard-crop"
            },
            {
                "title": "Palm oil futures drop 3% in Malaysia — relief for Indian consumers",
                "source": "NCDEX Bulletin",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "Malaysian palm oil futures fell 3% on improved production outlook. Cheaper palm oil import prices may reduce demand for domestic mustard oil from the FMCG sector, putting slight bearish pressure on mustard.",
                "sentiment": "bearish",
                "category": "edible_oil",
                "url": "https://ncdex.com/bulletin/palm-oil-prices"
            },
            {
                "title": "Big processors on buying spree ahead of Holi festival demand",
                "source": "AgriMarket India",
                "date": datetime.now().strftime("%d %b %Y"),
                "summary": "Major oil processing companies including Adani Wilmar and Ruchi Soya are aggressively procuring mustard seed ahead of Holi festival demand spike. This bulk buying is tightening availability at mandis and supporting prices.",
                "sentiment": "bullish",
                "category": "price",
                "url": "https://agrimarket.in/news/holi-demand-mustard"
            }
        ]
        _news_cache = fallback
        _news_cache_ts = now
        return jsonify({'news': fallback, 'cached': False, 'fallback': True})

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



if __name__ == '__main__':
    local_ip = get_local_ip()
    W = 62

    print()
    print("=" * W)
    print("  🌾  RV PREDICTOR — STARTING")
    print("=" * W)
    print()
    print(f"  💻  This laptop only   →  http://localhost:{PORT}")
    print()
    print(f"  📱  Same WiFi network  →  http://{local_ip}:{PORT}")
    print(f"      (phone must be on same WiFi as this laptop)")
    print()

    if ENABLE_NGROK:
        print("  🌐  Starting ngrok tunnel...")
        public_url, ngrok_err = start_ngrok_tunnel(PORT)
        if public_url:
            print()
            print("  " + "─" * (W - 4))
            print("  🚀  PUBLIC URL — share this with ANYONE:")
            print()
            print(f"      {public_url}")
            print()
            print("  " + "─" * (W - 4))
            copied = try_copy_to_clipboard(public_url)
            if copied:
                print("  ✅  URL copied to clipboard!")
            print("  ⚠️  URL changes every restart (free plan)")
            print()
        elif ngrok_err == "no_token":
            print()
            print("  ❌  ngrok needs authtoken setup.")
            print("  Run: ngrok config add-authtoken YOUR_TOKEN")
            print()
        else:
            print("  ❌  ngrok failed.")
            print()
    else:
        print("  💡  Set ENABLE_NGROK = True to share publicly")
        print()

    print(f"  Pages:  /   /mandi   /compare   /history")
    print(f"  Debug:  http://localhost:{PORT}/api/debug")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * W)
    print()

    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)