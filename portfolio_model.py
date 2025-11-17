"""
Streamlit Frontend: Portfolio Rebalancer + Hold Evaluator

Run with:
    streamlit run portfolio_model.py

Requirements:
    pip install streamlit yfinance pandas plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta


# ==============================
# Helper functions: data
# ==============================

def get_latest_prices(tickers):
    """Return a dict {ticker: latest_close_price}."""
    prices = {}
    for t in tickers:
        data = yf.Ticker(t).history(period="1d")
        if data.empty:
            raise ValueError(f"No price data for {t}")
        prices[t] = float(data["Close"].iloc[-1])
    return prices


def get_history(ticker, start, end=None):
    """Get historical daily close prices between start and end (datetime.date or str)."""
    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    elif isinstance(end, date):
        end = end.strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No historical data for {ticker}")
    return data["Close"]


# ==============================
# Helper functions: portfolio
# ==============================

def auto_target_weights(tickers):
    """Equal-weight each ticker if no target_weights specified."""
    n = len(tickers)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in tickers}


# You can edit this set with the tickers you consider "core ETFs"
ETF_TICKERS = {"VOO", "SPY", "QQQ", "VTI", "VEA", "VWO"}


def equal_weight_strategy(tickers):
    """All tickers get the same weight."""
    return auto_target_weights(tickers)


def growth_tilt_strategy(tickers):
    """
    Example: 70% in individual stocks, 30% in ETFs.
    Stocks share the 70% equally, ETFs share the 30% equally.
    """
    stocks = [t for t in tickers if t not in ETF_TICKERS]
    etfs = [t for t in tickers if t in ETF_TICKERS]

    # If we can't detect stocks/ETFs, just fall back to equal weight
    if not stocks or not etfs:
        return auto_target_weights(tickers)

    weights = {}
    stock_share = 0.70
    etf_share = 0.30

    stock_w = stock_share / len(stocks)
    etf_w = etf_share / len(etfs)

    for t in stocks:
        weights[t] = stock_w
    for t in etfs:
        weights[t] = etf_w
    return weights


def defensive_tilt_strategy(tickers):
    """
    Example: 70% in ETFs (core), 30% in individual stocks.
    """
    stocks = [t for t in tickers if t not in ETF_TICKERS]
    etfs = [t for t in tickers if t in ETF_TICKERS]

    if not stocks or not etfs:
        return auto_target_weights(tickers)

    weights = {}
    stock_share = 0.30
    etf_share = 0.70

    stock_w = stock_share / len(stocks)
    etf_w = etf_share / len(etfs)

    for t in stocks:
        weights[t] = stock_w
    for t in etfs:
        weights[t] = etf_w
    return weights


def get_portfolio_value(portfolio, prices):
    value_positions = 0.0
    for ticker, pos in portfolio["positions"].items():
        value_positions += pos["shares"] * prices[ticker]
    total_value = value_positions + portfolio["cash"]
    return total_value


def get_current_weights(portfolio, prices):
    total_value = get_portfolio_value(portfolio, prices)
    weights = {}
    for ticker, pos in portfolio["positions"].items():
        weights[ticker] = (pos["shares"] * prices[ticker]) / total_value
    weights["cash"] = portfolio["cash"] / total_value
    return weights


def generate_rebalance_trades(portfolio, model, prices, min_trade_value=200.0):
    """
    Compare current weights to target weights.
    If deviation > drift_threshold, suggest BUY/SELL to move back to target.
    """
    tickers = list(portfolio["positions"].keys())
    target_weights = model.get("target_weights", {})

    # If user didn't define weights, auto equal-weight
    if not target_weights:
        target_weights = auto_target_weights(tickers)

    total_value = get_portfolio_value(portfolio, prices)
    current_weights = get_current_weights(portfolio, prices)
    drift_threshold = model["drift_threshold"]

    trades = []

    for ticker, target_w in target_weights.items():
        current_w = current_weights.get(ticker, 0.0)
        deviation = current_w - target_w  # +ve = overweight, -ve = underweight

        desired_value = target_w * total_value
        current_value = current_w * total_value

        # Overweight → sell
        if deviation > drift_threshold:
            value_to_sell = current_value - desired_value
            if value_to_sell > min_trade_value:
                shares_to_sell = value_to_sell / prices[ticker]
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": round(shares_to_sell, 2),
                    "approx_value": round(value_to_sell, 2),
                })

        # Underweight → buy
        elif deviation < -drift_threshold:
            value_to_buy = desired_value - current_value
            if value_to_buy > min_trade_value and value_to_buy <= portfolio["cash"]:
                shares_to_buy = value_to_buy / prices[ticker]
                trades.append({
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": round(shares_to_buy, 2),
                    "approx_value": round(value_to_buy, 2),
                })

    return trades, target_weights, current_weights


def total_return(price_series):
    return float(price_series.iloc[-1] / price_series.iloc[0] - 1.0)


def get_recent_returns_table(tickers, days=30):
    """
    Build a small table with last price and ~{days}d return for each ticker.
    Used for the 'tracker' style snapshot.
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    rows = []
    for t in tickers:
        try:
            prices = get_history(t, start=start.date(), end=end.date())
        except Exception:
            continue
        if len(prices) < 2:
            continue
        last_price = float(prices.iloc[-1])
        ret = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
        rows.append(
            {
                "Ticker": t,
                "Last Price": round(last_price, 2),
                f"{days}d Return %": round(ret * 100, 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Last Price", f"{days}d Return %"])
    return pd.DataFrame(rows)


def evaluate_hold_decision(ticker, benchmark, start_date, end_date=None):
    """
    Compare holding 'ticker' vs 'benchmark' over a period.
    Returns dict with returns and outperformance.
    """
    asset_prices = get_history(ticker, start=start_date, end=end_date)
    bench_prices = get_history(benchmark, start=start_date, end=end_date)

    r_asset = total_return(asset_prices)
    r_bench = total_return(bench_prices)

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "start_date": start_date,
        "end_date": end_date or date.today(),
        "asset_return": r_asset,
        "benchmark_return": r_bench,
        "outperformance": r_asset - r_bench,
    }


# ==============================
# Intrinsic value + P/E helpers (auto via EPS × P/E)
# ==============================

def get_eps_ttm_for_tickers(tickers):
    """
    Fetch EPS (TTM) from Yahoo Finance for each ticker.
    Returns {ticker: eps_ttm} where available.
    """
    eps_dict = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            eps_val = info.get("trailingEps")
            if eps_val is not None:
                eps_dict[t] = float(eps_val)
        except Exception:
            continue
    return eps_dict


def get_pe_ratios(tickers):
    """
    Fetch P/E ratio (Trailing PE) from Yahoo Finance.
    Returns {ticker: pe_ratio} where available.
    """
    pe_dict = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pe_val = info.get("trailingPE")
            if pe_val is not None:
                pe_dict[t] = float(pe_val)
        except Exception:
            continue
    return pe_dict


def classify_pe(pe):
    """
    Classify P/E ratio into Good / Medium / Bad.
    <15  → GOOD (undervalued zone / classic value range)
    15–25 → MEDIUM (around market average)
    >25  → BAD (expensive / growth priced)
    Returns (label, display_string_with_emoji).
    """
    if pe is None or pe <= 0:
        return "N/A", "⚪ N/A"

    if pe < 15:
        return "GOOD", "🟢 GOOD"
    elif pe <= 25:
        return "MEDIUM", "🟡 MEDIUM"
    else:
        return "BAD", "🔴 BAD"


def compute_intrinsic_from_eps_pe(eps_dict, pe_multiple):
    """
    Intrinsic value = EPS_TTM * P/E multiple
    Returns {ticker: intrinsic_value}
    """
    intrinsic = {}
    for t, eps in eps_dict.items():
        intrinsic[t] = eps * pe_multiple
    return intrinsic


def get_valuation_flags(prices, intrinsic_values, eps_dict=None, margin=0.20):
    """
    Compare intrinsic vs current price.
    margin = 0.20 means +/-20% band for 'fair value'.
    Returns:
      {ticker: {intrinsic, price, diff, signal, eps?}}
    """
    results = {}
    for t, price in prices.items():
        iv = intrinsic_values.get(t)
        if iv is None:
            continue
        diff = (iv - price) / price  # +ve = intrinsic > price (undervalued)

        if diff >= margin:
            signal = "UNDERVALUED"
        elif diff <= -margin:
            signal = "OVERVALUED"
        else:
            signal = "NEAR FAIR"

        results[t] = {
            "intrinsic": iv,
            "price": price,
            "diff": diff,
            "signal": signal,
        }
        if eps_dict is not None and t in eps_dict:
            results[t]["eps"] = eps_dict[t]
    return results


# ==============================
# UI helpers
# ==============================

def parse_positions_input(text):
    """
    Parse textarea like:
        AAPL, 10, 150
        MSFT, 5, 300
    into {ticker: {shares, cost_basis}}
    """
    positions = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        try:
            shares = float(parts[1])
        except ValueError:
            continue
        cost_basis = float(parts[2]) if len(parts) >= 3 else 0.0
        positions[ticker] = {"shares": shares, "cost_basis": cost_basis}
    return positions


def parse_weights_input(text):
    """
    Parse textarea like:
        AAPL, 0.4
        MSFT, 0.3
        VOO, 0.3
    into {ticker: weight}
    """
    weights = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        try:
            w = float(parts[1])
        except ValueError:
            continue
        weights[ticker] = w
    return weights


# ==============================
# Streamlit App
# ==============================

def main():
    st.set_page_config(page_title="Model Portfolio Tool", layout="wide")

    # --- Small style tweaks only ---
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # If you want the emoji back, change this line:
    # st.title("📊 Model Portfolio Rebalancer & Live Dashboard")
    st.title("Model Portfolio Rebalancer & Live Dashboard")
    st.markdown(
        "Use this app to:\n"
        "- Define your portfolio (cash + positions)\n"
        "- Choose weighting strategies\n"
        "- See BUY/SELL rebalance suggestions\n"
        "- Track recent performance with charts\n"
        "- Compare a holding vs a benchmark over time\n"
        "- Use auto intrinsic value (EPS × P/E) and company P/E ratios\n"
    )

    # ======================
    # Sidebar: portfolio setup
    # ======================
    st.sidebar.header("Portfolio Setup")

    default_positions_text = """AAPL, 10, 150
MSFT, 5, 300
VOO, 3, 400
"""

    positions_text = st.sidebar.text_area(
        "Positions (ticker, shares, cost_basis)",
        value=default_positions_text,
        help="One per line: TICKER, shares, cost_basis",
        height=150,
    )

    cash = st.sidebar.number_input(
        "Cash balance",
        min_value=0.0,
        value=5000.0,
        step=500.0,
        help="Uninvested cash in your portfolio currency.",
    )

    drift_threshold = st.sidebar.slider(
        "Drift threshold for rebalance (%)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="If a position's weight is off by more than this %, a trade is suggested.",
    ) / 100.0

    benchmark = st.sidebar.text_input(
        "Benchmark ticker for evaluation",
        value="VOO",
        help="Used to answer 'was it good to hold?'",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Intrinsic Value Model (auto)")

    pe_multiple = st.sidebar.number_input(
        "Target P/E multiple for intrinsic valuation",
        min_value=1.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        help="Intrinsic = EPS (TTM) × this P/E. EPS (TTM) is fetched from Yahoo Finance.",
    )
    valuation_margin = st.sidebar.slider(
        "Valuation margin for under/overvalued (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="How far intrinsic must differ from price to call it under/over-valued.",
    ) / 100.0

    # weighting strategies
    weights_mode = st.sidebar.radio(
        "Target weights / strategy",
        [
            "Equal weight (auto)",
            "Preset: Growth tilt (70% stocks / 30% ETFs)",
            "Preset: Defensive tilt (30% stocks / 70% ETFs)",
            "Custom (enter below)",
        ],
    )

    custom_weights_text = st.sidebar.text_area(
        "Custom target weights (ticker, weight)",
        value="",
        help="Example: AAPL, 0.4\nMSFT, 0.3\nVOO, 0.3",
        height=120,
    )

    positions = parse_positions_input(positions_text)
    tickers = list(positions.keys())

    if not tickers:
        st.warning("Please enter at least one valid position in the sidebar.")
        return

    # Decide target_weights based on the chosen strategy
    if weights_mode == "Equal weight (auto)":
        target_weights = equal_weight_strategy(tickers)
    elif weights_mode.startswith("Preset: Growth"):
        target_weights = growth_tilt_strategy(tickers)
    elif weights_mode.startswith("Preset: Defensive"):
        target_weights = defensive_tilt_strategy(tickers)
    elif weights_mode == "Custom (enter below)" and custom_weights_text.strip():
        target_weights = parse_weights_input(custom_weights_text)
    else:
        target_weights = equal_weight_strategy(tickers)

    model = {
        "drift_threshold": drift_threshold,
        "benchmark": benchmark,
        "target_weights": target_weights,
    }

    portfolio = {
        "cash": cash,
        "positions": positions,
    }

    # ======================
    # Fetch prices once for everything
    # ======================
    try:
        prices = get_latest_prices(tickers)
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return

    total_value = get_portfolio_value(portfolio, prices)
    current_weights = get_current_weights(portfolio, prices)
    trades, final_target_weights, _ = generate_rebalance_trades(
        portfolio, model, prices
    )

    # Intrinsic value: EPS (TTM) × P/E multiple
    eps_ttm = get_eps_ttm_for_tickers(tickers)
    pe_ratios = get_pe_ratios(tickers)

    intrinsic_values = compute_intrinsic_from_eps_pe(eps_ttm, pe_multiple)
    valuation_flags = get_valuation_flags(
        prices,
        intrinsic_values,
        eps_dict=eps_ttm,
        margin=valuation_margin,
    )

    # ======================
    # Tabs: Dashboard / Rebalance / Backtest
    # ======================
    tab_dash, tab_rebal, tab_back = st.tabs(
        ["📈 Dashboard", "🔁 Rebalance", "⏱ Backtest"]
    )

    # --------- DASHBOARD TAB ----------
    with tab_dash:
        st.subheader("Portfolio Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total value", f"${total_value:,.2f}")
        c2.metric(
            "Cash",
            f"${portfolio['cash']:,.2f}",
            f"{current_weights.get('cash', 0.0)*100:.1f}% of portfolio",
        )
        c3.metric("Number of positions", len(tickers))

        # Allocation pie chart (including CASH)
        st.markdown("### Allocation by Asset")

        weights_rows = [
            {"Ticker": t, "Weight %": round(current_weights.get(t, 0.0) * 100, 2)}
            for t in tickers
        ]
        cash_weight = current_weights.get("cash", 0.0)
        if cash_weight > 0:
            weights_rows.append(
                {"Ticker": "CASH", "Weight %": round(cash_weight * 100, 2)}
            )

        df_weights = pd.DataFrame(weights_rows)

        fig = px.pie(
            df_weights,
            names="Ticker",
            values="Weight %",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            pull=[
                0.05 if row["Ticker"] == "CASH" else 0
                for _, row in df_weights.iterrows()
            ],
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recent performance "tracker"
        st.markdown("### Recent Performance Snapshot (last 30 days)")
        df_recent = get_recent_returns_table(tickers, days=30)
        st.dataframe(df_recent, use_container_width=True, height=260)

        # Intrinsic value vs market price, with P/E ratio and rating
        st.markdown("### Intrinsic Value vs Market Price (EPS × P/E) & P/E Rating")
        if valuation_flags:
            val_rows = []
            for t, info in valuation_flags.items():
                pe_val = pe_ratios.get(t)
                pe_label, pe_display = classify_pe(pe_val)

                val_rows.append(
                    {
                        "Ticker": t,
                        "EPS (TTM)": round(info.get("eps", float("nan")), 2)
                        if "eps" in info
                        else "",
                        "P/E Ratio": f"{pe_val:.2f}" if pe_val else "N/A",
                        "P/E Rating": pe_display,  # uses emoji: 🟢 / 🟡 / 🔴
                        "Intrinsic Value": round(info["intrinsic"], 2),
                        "Market Price": round(info["price"], 2),
                        "Mispricing %": round(info["diff"] * 100, 2),
                        "Intrinsic Signal": info["signal"],
                    }
                )
            df_val = pd.DataFrame(val_rows)
            st.dataframe(df_val, use_container_width=True)
        else:
            st.caption(
                "Could not compute intrinsic values. EPS (TTM) may be missing for these tickers."
            )

        # Price chart for a selected ticker
        st.markdown("### Price Chart")
        col_left, col_right = st.columns([1, 3])
        with col_left:
            chart_ticker = st.selectbox("Ticker to chart", tickers)
            chart_window = st.selectbox(
                "Lookback window",
                ["1M", "3M", "6M", "1Y"],
                index=1,
            )
        days_lookup = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
        lookback_days = days_lookup[chart_window]
        start_chart = date.today() - timedelta(days=lookback_days)
        try:
            chart_series = get_history(chart_ticker, start=start_chart)
            with col_right:
                st.line_chart(chart_series, height=260)
        except Exception as e:
            st.warning(f"Could not load history for {chart_ticker}: {e}")

    # --------- REBALANCE TAB ----------
    with tab_rebal:
        st.subheader("Current Portfolio")

        rows = []
        for t, pos in positions.items():
            price = prices[t]
            value = pos["shares"] * price
            weight = current_weights.get(t, 0.0)
            rows.append(
                {
                    "Ticker": t,
                    "Shares": pos["shares"],
                    "Price": round(price, 2),
                    "Value": round(value, 2),
                    "Weight %": round(weight * 100, 2),
                }
            )
        rows.append(
            {
                "Ticker": "CASH",
                "Shares": "",
                "Price": "",
                "Value": round(cash, 2),
                "Weight %": round(current_weights.get("cash", 0.0) * 100, 2),
            }
        )
        df_portfolio = pd.DataFrame(rows)
        st.dataframe(df_portfolio, use_container_width=True)

        st.markdown(f"**Total portfolio value:** `${total_value:,.2f}`")

        st.markdown("### Target Weights")
        target_rows = [
            {"Ticker": t, "Target Weight %": round(w * 100, 2)}
            for t, w in final_target_weights.items()
        ]
        df_targets = pd.DataFrame(target_rows)
        st.dataframe(df_targets, use_container_width=True)

        st.markdown("### Suggested Rebalance Trades")
        if not trades:
            st.success("No trades suggested. All positions are within the drift threshold.")
        else:
            # Attach valuation and P/E rating if available
            for tr in trades:
                tkr = tr["ticker"]
                val = valuation_flags.get(tkr)
                pe_val = pe_ratios.get(tkr)

                if val:
                    tr["Intrinsic Signal"] = val["signal"]
                else:
                    tr["Intrinsic Signal"] = ""

                _, pe_display = classify_pe(pe_val)
                tr["P/E Rating"] = pe_display

            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True)
            st.info(
                "These are **suggested** trades based on your target weights, drift threshold, "
                "and valuation labels (EPS-based intrinsic + P/E rating, if data is available). "
                "They are not financial advice."
            )

    # --------- BACKTEST TAB ----------
    with tab_back:
        st.subheader("Was It Good to Hold? (Simple Backtest)")

        col1, col2, col3 = st.columns(3)
        with col1:
            eval_ticker = st.selectbox("Ticker to evaluate", tickers)
        with col2:
            start_date = st.date_input("Start date", value=date(2023, 1, 1))
        with col3:
            end_date = st.date_input("End date", value=date.today())

        if st.button("Run backtest"):
            try:
                result = evaluate_hold_decision(
                    eval_ticker, benchmark, start_date, end_date
                )
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    f"{eval_ticker} return",
                    f"{result['asset_return']*100:5.2f}%",
                )
                c2.metric(
                    f"{benchmark} return",
                    f"{result['benchmark_return']*100:5.2f}%",
                )
                c3.metric(
                    "Outperformance",
                    f"{result['outperformance']*100:5.2f}%",
                )
                st.caption(
                    "This is a simple total-return comparison (no dividends modeled explicitly). "
                    "Use it as a rough gauge, not precise performance accounting."
                )


if __name__ == "__main__":
    main()

