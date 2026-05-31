# Atmospheric muon direction regression using GNN (KM3NeT)

Regresja kierunku ruchu **mionu atmosferycznego** zarejestrowanego w detektorze **KM3NeT** za pomocą grafowej sieci neuronowej (GNN) w PyTorch Geometric. Całość znajduje się w notatniku [`muon_gnn_direction.ipynb`](muon_gnn_direction.ipynb), przeznaczonym do uruchomienia w **Google Colab z GPU**.

## Opis problemu

Naładowane cząstki przelatujące przez detektor emitują promieniowanie Czerenkowa rejestrowane przez fotopowielacze (PMT). Klasyczny algorytm **Jmuon** rekonstruuje z tych sygnałów kierunek toru cząstki. Celem projektu jest nauczenie modelu, który **koryguje** tę rekonstrukcję w stronę kierunku prawdziwego:

```
(reco_dir_x, reco_dir_y, reco_dir_z) = f(jmuon_dir_x, jmuon_dir_y, jmuon_dir_z, jmuon_likelihood)
```

- **Wejście (4 cechy / zdarzenie):** `jmuon_dir_x`, `jmuon_dir_y`, `jmuon_dir_z` (zrekonstruowany kierunek) oraz `jmuon_likelihood` (jakość dopasowania toru).
- **Cel:** prawdziwy kierunek `dir_x`, `dir_y`, `dir_z` (`true_direction`).
- **Dane:** `atm_muon.h5` (~139 tys. zdarzeń), odczyt `pd.read_hdf(path, "y")`.

Charakterystyka danych: kierunki to wektory jednostkowe, a miony atmosferyczne są zawsze schodzące w dół (`dir_z < 0`). Baseline (sam `jmuon_dir`) ma bardzo dobrą medianę błędu kątowego (~0,3°), ale **ciężki ogon** błędnych rekonstrukcji (95 percentyl ~36°). Głównym zadaniem modelu jest **ujarzmienie tego ogona**.

## Wykorzystana architektura

Zastosowano schemat zgodny z literaturą teleskopów neutrinowych (GraphNeT/**DynEdge**, ParticleNet): **jedno zdarzenie = jeden graf**, przetwarzany jako `węzły → bloki message-passing → global pooling → głowa MLP → kierunek`.

Ponieważ dostępne są tylko 4 zagregowane cechy na zdarzenie (brak danych per-hit), zastosowano **graf cech** (*feature-as-node*):

- **Węzeł = pojedyncza cecha zdarzenia** (4 węzły), wektor cech węzła: `[wartość, one-hot typu]`.
- **Krawędzie:** pełny graf `K4` + pętle własne.
- **Warstwy message-passing** (porównywane w tuningu): **EdgeConv** (DynEdge/ParticleNet), **GAT**, **GraphSAGE**, **GCN**.
- **Pooling:** `global_mean_pool` + `global_max_pool`.
- **Głowa:** MLP → 3 wyjścia (kierunek, normalizowany do wektora jednostkowego), opcjonalnie dodatkowe wyjście `kappa` przy stracie von Mises–Fishera.
- **Regularyzacja:** Dropout, `weight_decay` (L2), BatchNorm, **early stopping**, scheduler LR.

**Funkcje straty:** kątowa `1 - cos Δθ`, **von Mises–Fisher** (z niepewnością `kappa`) oraz mieszanka `kąt + 0.05·vMF`.

**Tuning (Optuna):** `lr`, `hidden`, liczba bloków, dropout, `weight_decay`, typ warstwy, typ poolingu, typ straty; wybór modelu po walidacyjnej medianie błędu kątowego.

## Metryki ewaluacyjne

- **Rozdzielczość kątowa w percentylach** (mediana, 68%, 95%) oraz frakcja błędów > 5° — główna metryka fizyczna.
- **Histogramy residuów** (per składowa `x/y/z` oraz residuum kątowe).
- **Histogram pull** z nałożonym rozkładem Gaussa (kalibracja niepewności; przy vMF `sigma = 1/√kappa`).
- **Residua vs predykcja**.
- **Q-Q plot** residuów (porównanie z rozkładem normalnym).
- **MAE vs true** (błąd binowany po wartości prawdziwej).
- **Pred vs true** (z linią `y = x` i `R²`).
- **Krzywe uczenia** (strata train/val, błąd kątowy, LR).
- **Porównanie z baseline** `jmuon_dir` (rozkłady i CDF błędu kątowego).
