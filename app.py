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


@app.route('/api/status')
def api_status():
    """Health check endpoint for Render."""
    has_data = os.path.exists("data/today_prediction.json")
    return jsonify({
        "status": "ok",
        "has_prediction_data": has_data,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


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
        if not os.path.exists(json_path):
            return jsonify({"error": "today_prediction.json not found. Run daily_predict.py first.",
                            "data": [], "report_date": None})
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Get real data date from JSON or file mtime
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
            import os as _os
            mtime = _os.path.getmtime(json_path)
            data_date_str = datetime.fromtimestamp(mtime).strftime("%d %b %Y")

        def _s(v, default=0):
            if isinstance(v, str): return v
            try:
                return default if (v is None or (isinstance(v, float) and v != v)) else round(float(v), 2)
            except Exception: return default

        normalized = []
        for row in raw_data:
            normalized.append({
                "mandi":   _s(row.get("Mandi",           ""), ""),
                "price":   _s(row.get("Current Price",   0)),
                "pred1":   _s(row.get("Pred 1d ₹",  0)),
                "pred3":   _s(row.get("Pred 3d ₹",  0)),
                "pred7":   _s(row.get("Pred 7d ₹",  0)),
                "pred14":  _s(row.get("Pred 14d ₹", 0)),
                "ret1":    _s(row.get("Ret 1d %",        0)),
                "ret3":    _s(row.get("Ret 3d %",        0)),
                "ret7":    _s(row.get("Ret 7d %",        0)),
                "ret14":   _s(row.get("Ret 14d %",       0)),
                "conf1":   _s(row.get("Conf 1d %",       0)),
                "conf3":   _s(row.get("Conf 3d %",       0)),
                "conf7":   _s(row.get("Conf 7d %",       0)),
                "conf14":  _s(row.get("Conf 14d %",      0)),
                "score":   _s(row.get("Procurement Score", 0)),
                "crash":   _s(row.get("Crash Risk %",    0)),
                "entry":   _s(row.get("Entry Signal",    ""), ""),
                "plan":    _s(row.get("Planning Signal", ""), ""),
                "arrival": _s(row.get("Arrival Pressure", 0)),
                "zone":    _s(row.get("Price Zone",      ""), ""),
                "regime":  _s(row.get("Regime", row.get("Market Regime", "")), ""),
                "trend":   _s(row.get("Trend Shape",     ""), ""),
                "rows":    _s(row.get("Rows",             0)),
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
        json_path = "data/today_prediction.json"
        if not os.path.exists(json_path):
            return jsonify([])
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        result = []
        for row in raw_data:
            # Support both capitalized keys (from daily_predict.py) and lowercase
            mandi = str(row.get('Mandi',         row.get('mandi', ''))).strip()
            crash = float(row.get('Crash Risk %', row.get('crash', 0)) or 0)
            score = float(row.get('Procurement Score', row.get('score', 0)) or 0)
            price = float(row.get('Current Price', row.get('price', 0)) or 0)
            entry = str(row.get('Entry Signal',   row.get('entry', '')))

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
        if not os.path.exists("data/today_prediction.json"):
            return jsonify({"error": "No data yet"})
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
            "date": datetime.fromtimestamp(os.path.getmtime("data/today_prediction.json")).strftime("%d %b %Y") if os.path.exists("data/today_prediction.json") else datetime.now().strftime("%d %b %Y")
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


# ── API: MARKET NEWS via NewsAPI.org ──────────────────────────

NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY', '')

_news_cache    = {}   # keyed by cache_key
_news_cache_ts = {}
NEWS_CACHE_SECONDS = 300   # 5 min — fresh per category/query/page

# Each category has a DISTINCT query so switching pills gives different articles
_CATEGORY_QUERIES = {
    'all':        'mustard sarson edible oil India',
    'price':      'mustard seed mandi price arrivals India',
    'edible_oil': 'mustard oil soybean oil palm oil retail wholesale India',
    'export':     'edible oil import export duty India oilseed trade',
    'policy':     'mustard MSP minimum support price NCDEX government India',
    'weather':    'mustard rabi crop harvest rainfall Rajasthan India',
}

# Rotate sortBy so successive refreshes give different ordering of articles
# publishedAt → latest first, popularity → most-read first, relevancy → best match
_SORT_ROTATION = ['publishedAt', 'popularity', 'relevancy']
_sort_idx = 0


def _fetch_newsapi(query: str, page: int = 1,
                   page_size: int = 12, sort_by: str = 'publishedAt') -> list:
    """
    Calls NewsAPI /v2/everything.
    sort_by rotates between calls so Refresh gives visibly different ordering.
    """
    import urllib.request, urllib.parse
    from datetime import timedelta

    # Search window: last 30 days (free plan max is 1 month)
    date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    params = urllib.parse.urlencode({
        'q':        query,
        'from':     date_from,
        'sortBy':   sort_by,
        'page':     page,
        'pageSize': page_size,
        'language': 'en',
        'apiKey':   NEWSAPI_KEY,
    })
    url = f'https://newsapi.org/v2/everything?{params}'

    req = urllib.request.Request(url, headers={'User-Agent': 'RVPredictor/1.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    if data.get('status') != 'ok':
        raise ValueError(f"NewsAPI error: {data.get('message', data.get('status'))}")

    articles = []
    for a in data.get('articles', []):
        title = (a.get('title') or '').strip()
        if not title or title == '[Removed]':
            continue
        source_name = (a.get('source') or {}).get('name') or 'News'
        pub = a.get('publishedAt', '')
        try:
            dt       = datetime.strptime(pub[:10], '%Y-%m-%d')
            date_str = dt.strftime('%d %b %Y')
        except Exception:
            date_str = datetime.now().strftime('%d %b %Y')

        articles.append({
            'title':    title[:120],
            'source':   source_name,
            'date':     date_str,
            'summary':  (a.get('description') or '')[:280],
            'url':      a.get('url', ''),
            'image':    a.get('urlToImage') or '',
            'category': 'general',
        })
    return articles


def _fallback_news(today_str: str) -> list:
    return [
        {'title': 'Mustard prices firm at Jaipur mandi on tight supply',
         'source': 'AgriMarket India', 'date': today_str,
         'summary': 'Mustard seed prices remained firm at major Rajasthan mandis amid tight supply. Arrivals were down 15% week-on-week.',
         'url': '', 'image': '', 'category': 'price'},
        {'title': 'Edible oil imports rise — pressure on domestic mustard oil',
         'source': 'Economic Times Agri', 'date': today_str,
         'summary': 'India edible oil imports rose, led by palm oil. Cheaper imports are putting pressure on domestic mustard oil margins.',
         'url': '', 'image': '', 'category': 'export'},
        {'title': 'Rabi mustard crop area up in Rajasthan this season',
         'source': 'Rajasthan Krishi News', 'date': today_str,
         'summary': 'Mustard sowing area rose this rabi season. Bumper harvest expected in March-April.',
         'url': '', 'image': '', 'category': 'weather'},
        {'title': 'Mustard MSP provides price floor — market trading above support',
         'source': 'NCDEX Bulletin', 'date': today_str,
         'summary': 'MSP for mustard gives farmers a guaranteed floor price. Market prices are above MSP.',
         'url': '', 'image': '', 'category': 'policy'},
    ]


@app.route('/api/news')
def api_news():
    """
    Live mustard/oilseed news from NewsAPI.org.

    ?category=all|price|edible_oil|export|policy|weather  (default: all)
    ?q=custom search query    (overrides category)
    ?page=N                   (default 1, max ~8 on free plan)
    ?refresh=1                (bypass cache + rotate sort order → different results)
    """
    global _news_cache, _news_cache_ts, _sort_idx

    now       = datetime.now().timestamp()
    today_str = datetime.now().strftime('%d %b %Y')

    category      = request.args.get('category', 'all')
    custom_q      = request.args.get('q', '').strip()
    page          = max(1, int(request.args.get('page', 1) or 1))
    force_refresh = request.args.get('refresh', '0') == '1'

    # Build query string
    if custom_q:
        query = custom_q + ' India mustard oilseed'
    else:
        query = _CATEGORY_QUERIES.get(category, _CATEGORY_QUERIES['all'])

    # On force_refresh: rotate sort order so results genuinely differ
    if force_refresh:
        _sort_idx = (_sort_idx + 1) % len(_SORT_ROTATION)
    sort_by = _SORT_ROTATION[_sort_idx]

    # Cache key includes sort_by so rotate gives a new cache slot
    cache_key = f'{query}|{page}|{sort_by}'

    # Serve cache on normal loads
    if (not force_refresh
            and cache_key in _news_cache
            and (now - _news_cache_ts.get(cache_key, 0)) < NEWS_CACHE_SECONDS):
        age = int((now - _news_cache_ts[cache_key]) / 60)
        rem = max(0, int((NEWS_CACHE_SECONDS - (now - _news_cache_ts[cache_key])) / 60))
        return jsonify({
            'news':         _news_cache[cache_key],
            'cached':       True,
            'cache_age':    age,
            'next_refresh': rem,
            'query':        query,
            'sort_by':      sort_by,
            'page':         page,
            'count':        len(_news_cache[cache_key]),
        })

    if not NEWSAPI_KEY:
        fb = _fallback_news(today_str)
        return jsonify({'news': fb, 'cached': False, 'fallback': True,
                        'error': 'No NEWSAPI_KEY set — add it to Render environment variables'})

    try:
        print(f'[api/news] page={page} sort={sort_by} query="{query}"')
        articles = _fetch_newsapi(query, page=page, page_size=12, sort_by=sort_by)

        _news_cache[cache_key]    = articles
        _news_cache_ts[cache_key] = now

        print(f'[api/news] ✅ {len(articles)} articles')
        return jsonify({
            'news':     articles,
            'cached':   False,
            'sort_by':  sort_by,
            'query':    query,
            'page':     page,
            'count':    len(articles),
        })

    except Exception as e:
        print(f'[api/news] Error: {e}')
        # Return stale cache if available
        stale = _news_cache.get(cache_key)
        if stale:
            return jsonify({'news': stale, 'cached': True, 'stale': True,
                            'error': str(e), 'count': len(stale)})
        fb = _fallback_news(today_str)
        return jsonify({'news': fb, 'cached': False, 'fallback': True, 'error': str(e)})


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