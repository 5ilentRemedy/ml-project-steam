import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ustawienia dla wykresów
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Ładuje wyczyszczone dane"""
    data_path = Path(__file__).parent / "data" / "games_cleaned.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Załadowano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df

def prepare_numeric_data(df):
    """Przygotowuje dane numeryczne do analizy"""
    numeric_columns = [
        'Price', 'Metacritic score', 'Achievements', 'User score',
        'Score rank', 'Positive', 'Negative', 'Estimated owners', 'Review_ratio'
    ]

    # Konwertuj kolumny na numeryczne
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Usuń wiersze z brakami w kluczowych kolumnach
    df_clean = df.dropna(subset=['Price', 'Positive', 'Negative']).copy()

    print(f"Dane po oczyszczeniu: {df_clean.shape[0]} wierszy")
    return df_clean, numeric_columns

def create_histograms(df, numeric_columns):
    """Tworzy histogramy dla kolumn numerycznych"""
    print("\nGenerowanie histogramów...")

    # Wybierz kolumny do histogramów (bez bardzo rzadkich)
    hist_columns = ['Price', 'Positive', 'Negative', 'User score', 'Metacritic score']

    n_cols = 2
    n_rows = (len(hist_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, col in enumerate(hist_columns):
        if col in df.columns and i < len(axes):
            ax = axes[i]

            # Filtruj dane dla lepszej wizualizacji
            if col == 'Price':
                data = df[df[col] <= 100][col]  # Ogranicz do 100 USD
            elif col in ['Positive', 'Negative']:
                data = df[df[col] <= 10000][col]  # Ogranicz recenzje
            else:
                data = df[col].dropna()

            if len(data) > 0:
                sns.histplot(data=data, ax=ax, kde=True, bins=50)
                ax.set_title(f'Histogram {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Liczba gier')
                ax.grid(True, alpha=0.3)

    # Usuń puste wykresy
    for i in range(len(hist_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('reports/figures/histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Histogramy zapisane jako: reports/figures/histograms.png")

def create_correlation_matrix(df, numeric_columns):
    """Tworzy macierz korelacji"""
    print("\nGenerowanie macierzy korelacji...")

    # Wybierz tylko istniejące kolumny numeryczne
    available_numeric = [col for col in numeric_columns if col in df.columns]

    # Oblicz korelację
    corr_matrix = df[available_numeric].corr()

    # Utwórz heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.title('Macierz korelacji między zmiennymi', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Macierz korelacji zapisana jako: reports/figures/correlation_matrix.png")

def create_scatter_plots(df):
    """Tworzy wykresy punktowe dla kluczowych zależności"""
    print("\nGenerowanie wykresów punktowych...")

    # Wybrane pary zmiennych do analizy
    scatter_pairs = [
        ('Price', 'Positive'),
        ('Price', 'User score'),
        ('Positive', 'Negative'),
        ('User score', 'Metacritic score')
    ]

    n_cols = 2
    n_rows = (len(scatter_pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, (x_col, y_col) in enumerate(scatter_pairs):
        if i < len(axes) and x_col in df.columns and y_col in df.columns:
            ax = axes[i]

            # Filtruj dane dla lepszej wizualizacji
            plot_data = df[[x_col, y_col]].dropna()
            if x_col == 'Price':
                plot_data = plot_data[plot_data[x_col] <= 100]
            if x_col in ['Positive', 'Negative'] or y_col in ['Positive', 'Negative']:
                plot_data = plot_data[(plot_data[x_col] <= 10000) & (plot_data[y_col] <= 10000)]

            if len(plot_data) > 0:
                sns.scatterplot(data=plot_data, x=x_col, y=y_col, ax=ax, alpha=0.6)
                ax.set_title(f'Zależność: {x_col} vs {y_col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)

    # Usuń puste wykresy
    for i in range(len(scatter_pairs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('reports/figures/scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Wykresy punktowe zapisane jako: reports/figures/scatter_plots.png")

def create_boxplots(df, numeric_columns):
    """Tworzy boxploty dla wykrywania wartości odstających"""
    print("\nGenerowanie boxplotów (wartości odstające)...")

    # Wybierz kolumny do boxplotów
    box_columns = ['Price', 'Positive', 'Negative', 'User score', 'Metacritic score']

    n_cols = 2
    n_rows = (len(box_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, col in enumerate(box_columns):
        if col in df.columns and i < len(axes):
            ax = axes[i]

            # Filtruj dane dla lepszej wizualizacji
            if col == 'Price':
                data = df[df[col] <= 100][col]
            elif col in ['Positive', 'Negative']:
                data = df[df[col] <= 10000][col]
            else:
                data = df[col].dropna()

            if len(data) > 0:
                sns.boxplot(data=data, ax=ax, orient='h')
                ax.set_title(f'Boxplot {col} (wartości odstające)', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.grid(True, alpha=0.3)

    # Usuń puste wykresy
    for i in range(len(box_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('reports/figures/boxplots_outliers.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Boxploty zapisane jako: reports/figures/boxplots_outliers.png")

def detect_outliers(df, column, method='iqr'):
    """Wykrywa wartości odstające"""
    if column not in df.columns:
        return []

    data = df[column].dropna()

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    else:
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > 3]

    return outliers

def print_outlier_summary(df, numeric_columns):
    """Drukuje podsumowanie wartości odstających"""
    print("\n" + "="*80)
    print("PODSUMOWANIE WARTOŚCI ODSTAJĄCYCH")
    print("="*80)

    for col in ['Price', 'Positive', 'Negative', 'User score', 'Metacritic score']:
        if col in df.columns:
            outliers = detect_outliers(df, col)
            total = len(df[col].dropna())
            outlier_pct = (len(outliers) / total * 100) if total > 0 else 0

            print(f"\n{col}:")
            print(f"  - Łącznie wartości: {total}")
            print(f"  - Wartości odstające: {len(outliers)} ({outlier_pct:.1f}%)")
            if len(outliers) > 0:
                print(f"  - Zakres odstających: {outliers.min():.2f} - {outliers.max():.2f}")

def main():
    print("="*80)
    print("ANALIZA DANYCH - HISTOGRAMY, KORELACJE I WARTOŚCI ODSTAJĄCE")
    print("="*80)

    try:
        # Załaduj dane
        df = load_data()

        # Przygotuj dane numeryczne
        df_clean, numeric_columns = prepare_numeric_data(df)

        # Utwórz katalog na wykresy
        figures_dir = Path(__file__).parent / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Generuj wizualizacje
        create_histograms(df_clean, numeric_columns)
        create_correlation_matrix(df_clean, numeric_columns)
        create_scatter_plots(df_clean)
        create_boxplots(df_clean, numeric_columns)

        # Podsumowanie outlierów
        print_outlier_summary(df_clean, numeric_columns)

        print("\n" + "="*80)
        print("[OK] ANALIZA WIZUALNA UKOŃCZONA")
        print("Wykresy zapisane w katalogu: reports/figures/")
        print("="*80)

    except Exception as e:
        print(f"Błąd podczas analizy: {e}")
        raise

if __name__ == "__main__":
    main()