# Quallamaggie (Kristjan Kullamägi) Trading Strategy

## Executive Summary

Kristjan Kullamägi, known as "Qullamaggie," is a legendary Swedish day/swing trader who turned a small account into tens of millions using a systematic **momentum breakout strategy**. His approach focuses on capturing explosive moves in leading stocks through precise pattern recognition, disciplined entries, and asymmetric risk-reward management.

---

## Core Philosophy

> "I'm looking for stocks that can go up 20%, 50%, 100% or more. I'm willing to take many small losses to catch those big winners."

**Key Principles:**
- **Low win rate, massive winners**: Expects 25-35% win rate, but winners can be 10R-50R+
- **Momentum begets momentum**: Focus on stocks already showing relative strength
- **Patience over activity**: Most time spent in cash waiting for perfect setups
- **Pattern recognition over prediction**: Trade what you see, not what you think

---

## Strategy Components

### 1. Universe Selection (Stock Screening)

| Criteria | Requirement |
|----------|-------------|
| **Performance Rank** | Top 7% performers over 1, 3, and 6 months |
| **Price Action** | Above 50 SMA and 200 SMA (rising) |
| **Relative Strength** | RS Line at or near 52-week highs |
| **Volume Profile** | Above average daily volume (liquidity) |
| **Market Cap** | Mid-cap to large-cap preferred for liquidity |

**Screening Process:**
1. Scan for top momentum stocks (top 7% by 1/3/6-month returns)
2. Filter for stocks with RS Line at new highs
3. Look for recent surge (30-100% in 1-3 months)
4. Wait for consolidation pattern to form

---

### 2. Chart Patterns & Setups

Quallamaggie focuses on **four primary patterns** that occur after a significant price surge:

#### A. High Tight Flag (HTF)
- **Setup**: 90-100%+ surge in 1-8 weeks
- **Consolidation**: 10-25% pullback maximum
- **Duration**: 1-5 weeks of tight sideways action
- **Volume**: Drying up during consolidation
- **Entry**: Break above flag resistance on volume

#### B. VCP (Volatility Contraction Pattern)
- **Setup**: Series of higher lows with tightening range
- **Contractions**: 3-6 price contractions from left to right
- **Volume**: Decreases with each contraction
- **Pivot Point**: Final contraction creates tight pivot
- **Entry**: Break above pivot on expanding volume

#### C. Flat Base / Cup & Handle
- **Setup**: Rounded or flat consolidation base
- **Duration**: 2-8 weeks typically
- **Depth**: 15-35% pullback from highs
- **Handle**: Small 5-15% pullback near highs
- **Entry**: Break above handle resistance

#### D. Episodic Pivot (EP)
- **Setup**: Gap up of 10%+ on massive volume
- **Catalyst**: Earnings beat, major news, contract win
- **Volume**: 3-10x average volume
- **Action**: Stock holds/builds on initial gap
- **Entry**: Break of opening range high (ORH)

---

### 3. Entry Criteria (Opening Range High - ORH)

**Entry Signal**: Price breaks the high of a specified opening timeframe

| Timeframe | When to Use |
|-----------|-------------|
| **1-minute ORH** | Aggressive entries, episodic pivots |
| **5-minute ORH** | Standard breakout entries |
| **30-minute ORH** | More conservative, fewer shakeouts |
| **60-minute ORH** | Confirmed breakouts with lower risk |

**Entry Checklist:**
- [ ] Pattern is complete (flag, VCP, or EP)
- [ ] Price breaking above pivot point
- [ ] Volume expanding on breakout (1.5x+ average)
- [ ] Market conditions favorable (uptrend or neutral)
- [ ] RS Line confirming (at or near highs)

---

### 4. Stop Loss & Risk Management

#### Initial Stop Placement

| Rule | Description |
|------|-------------|
| **Low of Day (LOD)** | Primary stop: low of the breakout day |
| **ATR-Based** | Stop should be < 1 ATR ideally |
| **ADR Check** | Stop = 1/3 to 1/2 of ADR is "high star" setup |

#### Position Sizing

```
Position Size = (Account * Risk %) / (Entry - Stop)

Example:
- Account: $100,000
- Risk per trade: 0.5% = $500
- Entry: $50.00
- Stop: $48.00 (4% risk)
- Position Size: $500 / $2.00 = 250 shares
- Position Value: 250 × $50 = $12,500 (12.5% of account)
```

**Risk Parameters:**

| Parameter | Conservative | Standard | Aggressive |
|-----------|--------------|----------|------------|
| **Risk per Trade** | 0.25% | 0.5% | 1.0% |
| **Max Position** | 15% | 20% | 25% |
| **Max Exposure** | 100% | 150% | 200% (margin) |
| **Max Correlated** | 30% | 40% | 50% |

---

### 5. Trade Management & Exits

#### Scaling Out Strategy

| Phase | Action | Timing |
|-------|--------|--------|
| **Phase 1** | Sell 1/3 to 1/2 of position | Day 3-5 |
| **Phase 2** | Move stop to breakeven | After Phase 1 |
| **Phase 3** | Trail with moving average | Remaining position |

#### Trailing Stop Rules

| Stock Speed | Moving Average | Exit Signal |
|-------------|----------------|-------------|
| **Fast Mover** | 10-day EMA | Close below 10 EMA |
| **Normal** | 20-day EMA | Close below 20 EMA |
| **Slow Grind** | 50-day SMA | Close below 50 SMA |

> **Critical Rule**: Exit on a **CLOSE** below the MA, not intraday touches (avoiding shakeouts)

---

### 6. Technical Indicators

| Indicator | Purpose | Settings |
|-----------|---------|----------|
| **10 EMA** | Fast trailing stop, trend confirmation | Exponential |
| **20 EMA** | Standard trailing stop | Exponential |
| **50 SMA** | Intermediate trend, support | Simple |
| **200 SMA** | Long-term trend filter | Simple |
| **RS Line** | Relative strength vs S&P 500 | Custom |
| **Volume** | Breakout confirmation | 50-day average |

---

### 7. Fundamental "Rocket Fuel"

While technicals drive entries, fundamentals provide conviction:

| Metric | Ideal | Acceptable |
|--------|-------|------------|
| **EPS Growth** | 100%+ | 50%+ |
| **Revenue Growth** | 100%+ | 25%+ |
| **Estimates** | Upward revisions | Stable |
| **Theme/Sector** | Hot theme (AI, EV, etc.) | Strong sector |
| **Institutional** | Accumulation | Neutral |

---

## Expected Performance Metrics

Based on documented track record and strategy characteristics:

| Metric | Expected Range |
|--------|----------------|
| **Win Rate** | 25-35% |
| **Average Winner** | +15% to +30% |
| **Average Loser** | -3% to -7% |
| **Win/Loss Ratio** | 3:1 to 5:1 |
| **Profit Factor** | 2.0 - 4.0 |
| **Max Drawdown** | 15-30% |
| **Annual Return** | 50-200%+ (in good years) |
| **Sharpe Ratio** | 1.0 - 2.5 |

> **Note**: Returns are highly variable. Strategy performs exceptionally in momentum markets (2020, 2021), but struggles in choppy or bear markets.

---

## Implementation Checklist

- [ ] Set up momentum screening (top 7% by 1/3/6-month returns)
- [ ] Build RS Line indicator or use finviz.com
- [ ] Create watchlist management system
- [ ] Implement position sizing calculator
- [ ] Set up alerts for breakout candidates
- [ ] Study 1000+ historical breakout examples
- [ ] Paper trade for 3-6 months before live trading

---

## Key Differences from Other Strategies

| Aspect | Quallamaggie | Dual Momentum | HRP |
|--------|--------------|---------------|-----|
| **Asset Type** | Individual stocks | ETFs/Asset classes | ETFs/Stocks |
| **Timeframe** | Days to weeks | Months | Months |
| **Holding Period** | 3 days - 3 months | 6-12 months | Rebalance monthly |
| **Win Rate** | 25-35% | 50-60% | N/A (allocation) |
| **Focus** | Explosive moves | Trend following | Risk parity |
| **Skill Required** | High (discretionary) | Low (systematic) | Low (systematic) |
| **Time Commitment** | Daily monitoring | Monthly check | Monthly check |

---

## References

1. **YouTube**: [QULLAMAGGIE Stock Trading Strategy EXPLAINED](https://www.youtube.com/watch?v=we5LLjFlHCc)
2. **Qullamaggie Twitch Streams**: Live trading and education
3. **Mark Minervini**: SEPA methodology (similar approach)
4. **William O'Neil**: CAN SLIM (foundational concepts)
5. **Model Book Study**: Historical chart pattern analysis

---

## Risk Disclaimer

This strategy requires significant skill, discipline, and experience. The documented returns are exceptional and not typical. Most traders lose money attempting momentum strategies. Paper trade extensively before risking real capital.

---

## Version History

- **v1.0.0** (2024-12-25): Initial documentation from video analysis
