# analysis.py
import pandas as pd
import numpy as np

# --- WPROWADZONE ZMIANY ---
# POPRAWKA 5: Rozbudowa analizy o kluczowe metryki wydajności.
#   - Dodano obliczenia dla: Win Rate, Profit Factor, Max Drawdown i Sharpe Ratio.
#   - Funkcja `analyze_by_market_cap` została całkowicie przebudowana, aby przyjmować DataFrame z pojedynczymi transakcjami.
#   - Raport jest teraz znacznie bardziej szczegółowy i użyteczny.

def calculate_performance_metrics(pnl_series):
    """Oblicza zestaw metryk wydajności dla serii wyników PnL."""
    if pnl_series.empty or len(pnl_series) < 2:
        return {
            'Total Trades': len(pnl_series), 'Win Rate (%)': 0, 'Profit Factor': 0,
            'Sharpe Ratio': 0, 'Max Drawdown (%)': 0, 'Average PnL (%)': 0
        }
    
    # Obliczenia podstawowe
    total_trades = len(pnl_series)
    winning_trades = pnl_series[pnl_series > 0]
    losing_trades = pnl_series[pnl_series < 0]
    
    # Win Rate
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit Factor
    total_profit = winning_trades.sum()
    total_loss = abs(losing_trades.sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
    
    # Sharpe Ratio (zakładając stopę wolną od ryzyka = 0)
    sharpe_ratio = pnl_series.mean() / pnl_series.std() if pnl_series.std() > 0 else 0
    
    # Max Drawdown
    cumulative_pnl = (1 + pnl_series / 100).cumprod()
    peak = cumulative_pnl.expanding(min_periods=1).max()
    drawdown = (cumulative_pnl - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100
    
    return {
        'Total Trades': total_trades,
        'Win Rate (%)': f"{win_rate:.2f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown (%)': f"{max_drawdown:.2f}",
        'Average PnL (%)': f"{pnl_series.mean():.3f}",
        'Median PnL (%)': f"{pnl_series.median():.3f}"
    }

def analyze_by_market_cap(results_df, mcap_data):
    """Analizuje wyniki, grupując je po MCap i interwale, z rozszerzonymi metrykami."""
    if results_df.empty:
        print("Brak wyników do analizy.")
        return

    # Dołączanie danych o kapitalizacji rynkowej
    results_df['MarketCap'] = results_df['Pair'].map(mcap_data).fillna(0)
    
    # Definicja koszyków kapitalizacji rynkowej
    bins = [0, 1e9, 10e9, 200e9, float('inf')]
    labels = ['Small-Cap (<$1B)', 'Mid-Cap ($1B-$10B)', 'Large-Cap ($10B-$200B)', 'Mega-Cap (>$200B)']
    results_df['MCap_Bucket'] = pd.cut(results_df['MarketCap'], bins=bins, labels=labels, right=False)

    print("\n\n--- 📊 Końcowy, Zaawansowany Raport Wydajności Strategii ---")
    
    # Agregacja wyników
    grouped = results_df.groupby(['Timeframe', 'MCap_Bucket'])
    
    analysis_results = []
    for name, group in grouped:
        metrics = calculate_performance_metrics(group['PnL'])
        metrics['Timeframe'] = name[0]
        metrics['MCap_Bucket'] = name[1]
        analysis_results.append(metrics)
        
    if not analysis_results:
        print("Nie znaleziono wystarczających danych do wygenerowania raportu.")
        return

    final_report = pd.DataFrame(analysis_results)
    final_report = final_report.set_index(['Timeframe', 'MCap_Bucket'])
    
    print("\n--- Pełne metryki wydajności w zależności od Interwału i Kapitalizacji Rynkowej ---")
    print(final_report)

    # Zapis do pliku CSV
    final_report.to_csv("full_performance_report.csv")
    print("\n✅ Rozszerzony raport został zapisany do pliku 'full_performance_report.csv'")