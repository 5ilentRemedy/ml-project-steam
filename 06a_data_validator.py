"""
06a_data_validator.py - PEŁNA WALIDACJA STATYSTYCZNA I RAPORTOWANIE WYKRESÓW

Ten skrypt wczytuje dane wygenerowane przez FeatureEngineer i DataExporter,
przeprowadza ostateczną walidację i generuje 4 kluczowe wykresy dla klasyfikacji ML.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Ustawienia dla wykresów
plt.style.use('default')
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Ładuje wyczyszczone dane przetworzone w potoku"""
    try:
        data_path = Path(__file__).parent / "data" / "games_cleaned.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✓ Załadowano dane do walidacji: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
        return df
    except Exception as e:
        print(f"✗ Błąd podczas ładowania danych: {e}")
        raise

def prepare_numeric_data(df):
    """Przygotowuje ostateczne cechy numeryczne (zgodnie z 04_data_export.py)"""
    try:
        # Skupiamy się na cechach, które faktycznie wchodzą do końcowego modelu
        numeric_columns = [
            'Price', 'Release_year', 'Days_since_release', 'Platform_count', 
            'Total_reviews', 'Review_ratio', 'Log_owners', 'Log_total_reviews', 'Genre_count'
        ]

        # Konwertuj kolumny na numeryczne, jeśli to konieczne
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Usuwamy rzędy z krytycznymi brakami danych w zmiennej docelowej lub cenie
        df_clean = df.dropna(subset=['Price', 'Review_ratio', 'Market_Success']).copy()

        print(f"✓ Dane po ostatecznym dopasowaniu numerycznym: {df_clean.shape[0]} wierszy")
        return df_clean, numeric_columns
    except Exception as e:
        print(f"✗ Błąd podczas przygotowania danych: {e}")
        raise

def create_correlation_matrix(df, numeric_columns):
    """⭐ Wizualizacja 1/4: Macierz korelacji cech końcowych"""
    try:
        print("\n📊 Generowanie macierzy korelacji...")
        available_numeric = [col for col in numeric_columns if col in df.columns]
        corr_matrix = df[available_numeric].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt=".2f")

        plt.title('Macierz korelacji cech wejściowych modelu', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        figures_dir = Path(__file__).parent / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(figures_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Zapisano: correlation_matrix.png")
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia macierzy korelacji: {e}")

def create_top_correlations_barplot(df, numeric_columns):
    """⭐ Wizualizacja 2/4: Top 10 najsilniejszych korelacji liniowych"""
    try:
        print("\n📊 Generowanie wykresu najsilniejszych zależności...")
        available_numeric = [col for col in numeric_columns if col in df.columns]
        corr_matrix = df[available_numeric].corr()
        
        corr_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr'})
        )
        corr_pairs['abs_corr'] = corr_pairs['corr'].abs()
        top_corr = corr_pairs.sort_values('abs_corr', ascending=False).head(10)

        if top_corr.empty:
            return

        top_corr['pair'] = top_corr['feature_1'] + ' – ' + top_corr['feature_2']
        top_corr = top_corr.sort_values('abs_corr', ascending=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_corr, x='abs_corr', y='pair', palette='vlag', hue='pair', legend=False)
        plt.title('Top 10 najsilniejszych relacji liniowych', fontsize=12, fontweight='bold')
        plt.xlabel('Bezwzględna wartość współczynnika korelacji')
        plt.ylabel('Para cech')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        figures_dir = Path(__file__).parent / "reports" / "figures"
        plt.savefig(figures_dir / "top_correlations_barplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Zapisano: top_correlations_barplot.png")
        
        # Dodatkowo eksportujemy tekstowe streszczenie korelacji do pliku raportu json
        report_path = Path(__file__).parent / "reports" / "02_validation_report.json"
        summary_data = {"top_correlations": top_corr[['pair', 'corr']].to_dict(orient='records')}
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"✗ Błąd przy tworzeniu wykresu: {e}")

def create_market_success_countplot(df):
    """⭐ Wizualizacja 3/4: Liczebność klas sukcesu rynkowego (Zbalansowana)"""
    try:
        print("\n📊 Generowanie wykresu rozkładu sukcesu rynkowego...")
        plt.figure(figsize=(10, 5))
        
        order = ['Zrozumiały Hit', 'Ukryty Diament', 'Przeciętniak', 'Klapa rynkowa']
        sns.countplot(data=df, y='Market_Success', order=order, palette='Set2', hue='Market_Success', legend=False)
        
        plt.title('Zbalansowanie klas docelowych w oczyszczonym zbiorze danych', fontsize=13, fontweight='bold')
        plt.xlabel('Liczba gier zarejestrowanych w klasie')
        plt.ylabel('Klasa Sukcesu Rynkowego (Target)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        figures_dir = Path(__file__).parent / "reports" / "figures"
        plt.savefig(figures_dir / "market_success_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Zapisano: market_success_distribution.png")
    except Exception as e:
        print(f"✗ Błąd przy tworzeniu wykresu rozkładu klas: {e}")

def create_advanced_plots(df):
    """⭐ Wizualizacja 4/4: Zaawansowane wykresy relacji z podziałem na sukces rynkowy"""
    try:
        print("\n📊 Generowanie zaawansowanych wykresów relacji klasowych...")
        
        # Filtrujemy skrajne anomalie cenowe powyżej 100$ dla zachowania czytelności rozkładów
        plot_data = df[df['Price'] <= 100].copy()

        fig = plt.figure(figsize=(20, 6))
        hue_order = ['Zrozumiały Hit', 'Ukryty Diament', 'Przeciętniak', 'Klapa rynkowa']
        custom_palette = {'Zrozumiały Hit': '#2ecc71', 'Ukryty Diament': '#3498db', 'Przeciętniak': '#f1c40f', 'Klapa rynkowa': '#e74c3c'}

        # --- WYKRES 1: Cena vs Łączna liczba opinii (Skala Logarytmiczna) ---
        ax1 = fig.add_subplot(131)
        sns.scatterplot(data=plot_data, x='Price', y='Total_reviews', hue='Market_Success', 
                        hue_order=hue_order, palette=custom_palette, alpha=0.5, s=20, ax=ax1)
        ax1.set_title('Czy cena determinuje wolumen i sukces?', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Cena ($)')
        ax1.set_ylabel('Suma recenzji (Skala Log)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()

        # --- WYKRES 2: Rozkład Cen w Zależności od Klasy (Boxplot) ---
        ax2 = fig.add_subplot(132)
        sns.boxplot(data=plot_data, x='Market_Success', y='Price', order=hue_order, 
                    palette=custom_palette, ax=ax2, hue='Market_Success', legend=False)
        ax2.set_title('W jakich przedziałach cenowych leżą Hity i Diamenty?', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Klasa sukcesu rynkowego')
        ax2.set_ylabel('Cena ($)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
        ax2.grid(True, alpha=0.3)

        # --- WYKRES 3: Pozytywne vs Negatywne (Pokazuje separowalność geometryczną klas) ---
        ax3 = fig.add_subplot(133)
        active_reviews = plot_data[(plot_data['Positive'] > 0) & (plot_data['Negative'] > 0)]
        sns.scatterplot(data=active_reviews, x='Positive', y='Negative', hue='Market_Success',
                        hue_order=hue_order, palette=custom_palette, alpha=0.4, s=15, ax=ax3)
        ax3.set_title('Pozytywne vs Negatywne (Granice Decyzyjne Klas)', fontsize=11, fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Pozytywne recenzje (Log)')
        ax3.set_ylabel('Negatywne recenzje (Log)')
        ax3.grid(True, alpha=0.3)
        
        plt.legend(title='Klasa Sukcesu', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        figures_dir = Path(__file__).parent / "reports" / "figures"
        plt.savefig(figures_dir / "advanced_dependency_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Zapisano: advanced_dependency_plots.png")
        
    except Exception as e:
        print(f"✗ Błąd przy tworzeniu zaawansowanych wykresów: {e}")

def main():
    """Główna funkcja walidacji"""
    print("="*80)
    print("🔍 WALIDACJA DANYCH - ANALIZA ZBALANSOWANEGO SUKCESU RYNKOWEGO")
    print("="*80)
    
    try:
        # 1. Załaduj dane
        df = load_data()

        # 2. Przygotuj zmienne numeryczne do analizy
        df_clean, numeric_columns = prepare_numeric_data(df)

        # Utwórz katalog na wykresy
        figures_dir = Path(__file__).parent / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # 3. Wywołanie 4 kompletnych raportów graficznych
        print("\n🎯 Generowanie pełnego zestawu wizualizacji...")
        create_correlation_matrix(df_clean, numeric_columns)
        create_top_correlations_barplot(df_clean, numeric_columns)
        create_market_success_countplot(df_clean)
        create_advanced_plots(df_clean)

        print("\n" + "="*80)
        print("✓ WALIDACJA UKOŃCZONA POMYŚLNIE")
        print(f"  - Wszystkie poprawne wykresy i raporty zapisano w: reports/figures/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Błąd podczas walidacji: {e}")
        print("="*80 + "\n")
        raise

if __name__ == "__main__":
    main()