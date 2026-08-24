# Pomiar: memory probe DPO na stosie 4.5B + adapter D01 (2026-08-24)

Skrypt: `scripts/probe_task07_dpo_memory.py`, runtime: `src/doc2query/training/dpo_runtime.py`.
Artefakt: `artifacts/task07/dpo_memory_probe_v1/probe.json`.
Środowisko: `.venv-gpu` (torch 2.6.0+cu124), RTX 3060 Ti 8 GB, baza
`speakleash/Bielik-4.5B-v3.0-Instruct` w NF4, adapter `runs/D01-4.5B-STYLE-50K-S42/adapter`,
`HF_HUB_OFFLINE=1`.

**Czym to nie jest.** To pomiar wykonalności i kosztu na **syntetycznych** parach o
kontrolowanej długości. Nie użyto par v2.1 ani żadnych danych preferencyjnych, nie
otwarto testów finalnych, nie wybrano beta ani LR, nie ma tu wyniku selekcyjnego.
`task07_training_authorized=false` bez zmian; bramka V2.1-05 pozostaje nierozstrzygnięta.

## 1. Wynik: DPO 4.5B mieści się na 8 GB z dużym zapasem

| faza | `max_length=512` | `max_length=768` |
|---|---|---|
| załadowanie bazy 4-bit + adapter | 2,673 GiB | 2,688 GiB |
| precompute logprobów referencji (`no_grad`) | **2,710 GiB** | **2,779 GiB** |
| kroki treningowe (gradienty + AdamW, batch 1, gradient checkpointing) | **3,776 GiB** | **4,142 GiB** |
| czas precompute | 0,711 s/para | 0,861 s/para |
| czas treningu | 2,599 s/krok | 3,697 s/krok |

Dla porównania: SFT 4.5B (W06) miał peak **7,74 GiB** na tej samej karcie. DPO w tej
architekturze zużywa **o połowę mniej**, bo faza treningowa nie trzyma drugiego modelu —
referencja jest policzona wcześniej i wczytana jako dwie liczby na parę. Zapas do 8 GB
wynosi ~3,9 GiB przy `max_length=768`, więc batch > 1 albo dłuższy kontekst są
realne — ale to trzeba zmierzyć osobno, nie zakładać.

## 2. Ekstrapolacja kosztu na obecny zbiór 2 253 par

| | `max_length=512` | `max_length=768` |
|---|---|---|
| precompute całego zbioru | ~27 min | ~32 min |
| jedna epoka treningu (batch 1) | ~1,6 h | ~2,3 h |

Wniosek planistyczny: **sam DPO jest tani**. Przy trzech obowiązkowych ramionach
(DPO, continued SFT, score-weighted continued SFT) trening to rząd 5–7 h GPU, a
kosztem dominującym pozostaje **ewaluacja probe**: zmierzone ~1500 s na run, przy
wymaganych ≥5 parach seedów (sd par 0,0126 → półszerokość CI 0,0110 przy 5 parach,
0,0078 przy 10) i zmierzonym 18,5% odsetku zapadnięć wymagających reseedu. Trzy
ramiona × 5 seedów to ~6,3 h plus ~1,2 h na reseedy — czyli **ewaluacja jest droższa
od treningu**, i to ona wyznacza rozmiar okna GPU.

## 3. Znaleziska uboczne

- **Truncation działa i jest raportowany**: wszystkie 6 syntetycznych promptów (180
  słów pasażu) przekroczyło budżet przy obu długościach i zostały ucięte **od lewej**,
  po stronie promptu; completion pozostał nienaruszony. Peak pamięci jest więc
  zmierzony dokładnie na `max_length`, czyli w najgorszym przypadku, a nie na krótszych
  sekwencjach. Dla prawdziwych par trzeba osobno policzyć rozkład długości promptu
  (`validate_token_length_evidence` istnieje i czeka na dane).
- **Strata spada, a `reward_accuracy` nie jest degenerowana** (1,0 przy 512 i 0,75 przy
  768 na czterech krokach) — to sanity check mechaniki, nie wynik jakościowy: pary są
  syntetyczne, a referencja stała.
- **Referencja jest dowodliwie punktem startowym.** Precompute liczy logproby tym samym
  stosem (baza + adapter D01), który potem startuje jako polityka, więc wymóg Task 07
  („walidacja, że model referencyjny odpowiada dokładnie punktowi startowemu") jest
  spełniony konstrukcyjnie, a nie przez zaufanie do biblioteki. Wariant `trl.DPOTrainer`
  z `ref_model=None` przy modelu PEFT policzyłby referencję z **wyłączonym** adapterem,
  czyli z bazy — i to jest powód, dla którego runtime jest dwufazowy.

## 4. Czego nadal brakuje do runu DPO

1. **bramka V2.1-05** — brakuje 23 requestów u `gpt-oss`, czyli jednego okna dobowego;
2. **autoryzacja właściciela** (`task07_training_authorized=false` to osobna flaga);
3. **objętość danych** — 2 253 pary wobec ablacji zakładających 20k/50k/100k; pełna pula
   osi A to 15 989 par, ale kohorty v4–v11 są zamknięte do pozytywnej bramki;
4. handoff danych (preference i continued-SFT train/dev + wagi), token lengths na
   prawdziwych parach, podłączenie `doc2query train dpo` (nadal stub) i orkiestracja
   config → precompute → trening → manifest runu.

`task07_training_authorized=false`, `final_tests_used=[]`.
