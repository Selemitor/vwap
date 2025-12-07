# analysis.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re # Dodano import do czyszczenia nazw plików

def calculate_performance_metrics(pnl_series):
    if pnl_series.empty or len(pnl_series) < 2:
        return {
            'Total Trades': len(pnl_series), 'Win Rate (%)': 0, 'Profit Factor': 0,
            'Sharpe Ratio': 0, 'Max Drawdown (%)': 0, 'Average PnL (%)': 0
        }
    total_trades = len(pnl_series)
    winning_trades = pnl_series[pnl_series > 0]
    losing_trades = pnl_series[pnl_series < 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_profit = winning_trades.sum()
    total_loss = abs(losing_trades.sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
    sharpe_ratio = pnl_series.mean() / pnl_series.std() if pnl_series.std() > 0 else 0
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

def plot_equity_curve(results_df, title_suffix=""):
    """Rysuje krzywą kapitału dla wszystkich transakcji posortowanych chronologicznie."""
    if results_df.empty: return

    # Sortujemy transakcje po czasie wyjścia
    df = results_df.sort_values(by='exit_time').copy()
    
    # Zakładamy start z 100% kapitału. PnL jest w procentach (np. 5.0 to 5%)
    # Konwertujemy procenty na mnożniki (5% -> 1.05)
    df['multiplier'] = 1 + (df['PnL'] / 100.0)
    df['equity'] = df['multiplier'].cumprod() * 100 # Start od 100 jednostek

    plt.figure(figsize=(12, 6))
    plt.plot(df['exit_time'], df['equity'], label='Equity Curve', color='blue', linewidth=1.5)
    
    # Dodanie zaznaczenia Drawdownów
    rolling_max = df['equity'].cummax()
    # Fill between equity and rolling max to show drawdown areas clearly
    plt.fill_between(df['exit_time'], df['equity'], rolling_max, color='red', alpha=0.1, label='Drawdown')

    plt.title(f"Equity Curve - {title_suffix}")
    plt.xlabel("Data")
    plt.ylabel("Kapitał (start=100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- POPRAWKA: SANITYZACJA NAZWY PLIKU ---
    # Usuwamy znaki niedozwolone w Windows: < > : " / \ | ? *
    safe_suffix = title_suffix.replace(' ', '_')
    safe_suffix = re.sub(r'[<>:"/\\|?*]', '', safe_suffix) # Usuwa znaki specjalne
    
    filename = f"equity_curve_{safe_suffix}.png"
    plt.savefig(filename)
    print(f"📈 Zapisano wykres kapitału: {filename}")
    plt.close()

def analyze_by_market_cap(results_df, mcap_data):
    if results_df.empty:
        print("Brak wyników do analizy.")
        return

    results_df['MarketCap'] = results_df['Pair'].map(mcap_data).fillna(0)
    bins = [0, 1e9, 10e9, 200e9, float('inf')]
    labels = ['Small-Cap (<$1B)', 'Mid-Cap ($1B-$10B)', 'Large-Cap ($10B-$200B)', 'Mega-Cap (>$200B)']
    results_df['MCap_Bucket'] = pd.cut(results_df['MarketCap'], bins=bins, labels=labels, right=False)

    print("\n\n--- 📊 Końcowy, Zaawansowany Raport Wydajności Strategii ---")
    
    # Rysujemy ogólną krzywą kapitału dla całego portfela
    plot_equity_curve(results_df, "All_Portfolio")

    grouped = results_df.groupby(['Timeframe', 'MCap_Bucket'], observed=True) # observed=True dla kategorii
    
    analysis_results = []
    for name, group in grouped:
        if group.empty: continue # Pomijamy puste grupy
        
        metrics = calculate_performance_metrics(group['PnL'])
        metrics['Timeframe'] = name[0]
        metrics['MCap_Bucket'] = name[1]
        analysis_results.append(metrics)
        
        # Opcjonalnie: Rysuj krzywą dla każdej grupy osobno
        if len(group) > 20: # Rysuj tylko jeśli jest wystarczająco dużo transakcji
             plot_equity_curve(group, f"{name[0]}_{name[1]}")
        
    if not analysis_results:
        print("Nie znaleziono wystarczających danych do wygenerowania raportu.")
        return

    final_report = pd.DataFrame(analysis_results)
    final_report = final_report.set_index(['Timeframe', 'MCap_Bucket'])
    print(final_report)
    final_report.to_csv("full_performance_report.csv")
    print("\n✅ Rozszerzony raport zapisany.")

