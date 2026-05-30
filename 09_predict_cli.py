from steam_ml.core import MODEL_FEATURES, predict_from_values


PROMPTS = {
    "Release_year": "Rok wydania",
    "Days_since_release": "Dni od premiery",
    "Platform_count": "Liczba platform",
    "Price": "Cena USD",
    "Is_free": "Czy darmowa? 1/0",
    "Total_reviews": "Laczna liczba recenzji",
    "Review_ratio": "Udzial pozytywnych recenzji 0-1",
    "Log_owners": "Log wlascicieli, np. 10",
    "Has_achievements": "Czy ma osiagniecia? 1/0",
    "Log_total_reviews": "Log recenzji, np. 5",
    "Genre_count": "Liczba gatunkow",
}


def read_float(feature: str) -> float:
    raw = input(f"{PROMPTS.get(feature, feature)}: ").strip().replace(",", ".")
    return float(raw) if raw else 0.0


if __name__ == "__main__":
    print("Predykcja sukcesu gry Steam")
    values = {feature: read_float(feature) for feature in MODEL_FEATURES}
    result = predict_from_values(values)
    print(f"\nModel: {result['model']}")
    print(f"Wynik: {result['label']}")
    print(f"Prawdopodobienstwo wysokiej oceny: {result['probability']:.2%}")
