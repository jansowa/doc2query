# Audyt merytoryczny danych SFT: 12 000 par, cztery osie (2026-08-31)

## Status

Pomiar wykonany na serwerze inferencji właściciela (Qwen3.8-27B, temperature 0,
prompt `task06-night-jobs-pl-v1`), na losowej próbce **12 000 par** z puli
treningowej SFT (384 576 par, seed 20260830, zamrożony przed generacją).
Pomysł i autoryzacja: właściciel, 2026-08-30 („wydaje mi się, że da się jeszcze
jakieś dodatkowe filtry zrobić"). **To pomiar wad w istniejącej puli, nie
filtr** — żadna para nie została usunięta. `final_tests_used=[]`.

Kontekst: jedynym filtrem merytorycznym puli był dotąd próg rerankera
`source_en_score >= 23.50` (na tekstach angielskich, przed tłumaczeniem);
kontrole Task 01 są strukturalne (mojibake, HTML, długości, heurystyka
polskości, próg pokrycia zapytania).

## 1. Wynik

| oś | wadliwych | udział |
|---|---|---|
| `nieodpowiadalne` — pasaż nie zawiera odpowiedzi na zapytanie | 1 833 | **15,3%** |
| `zła polszczyzna` — kalki, resztki angielskiego, przekręcone nazwy | 1 388 | **11,6%** |
| `zbyt ogólne` | 282 | 2,4% |
| `niesensowne` | 84 | 0,7% |
| **co najmniej jedna wada twarda** (bez `zbyt ogólne`) | **2 979** | **24,8%** |

Główny problem wg sędziego: 75,0% `brak`, 11,6% `tłumaczenie`,
11,3% `nieodpowiadalne`, 2,1% `zbyt ogólne`.

Przykłady (dosłowne, z próbki):

- **tłumaczenie**: „jaki kolor ma biały koń georgeâa washingtona" (mojibake w
  nazwie własnej przeszło przez heurystyki), „co to jest malipu" (przekręcone
  Maltipoo), „gdzie jest carbonado wa" (surowy skrót stanu z EN).
- **nieodpowiadalne**: „czy koniczyna biskupia jest rodzima" przy pasażu o
  koniczynie pospolitej; „jak zrobić krzywą kalibracyjną" przy pasażu
  opisującym wyłącznie krok 3 użycia gotowej krzywej.

**Cała pula jest powyżej progu rerankera**: wszystkie 384 576 par ma
`source_en_score ≥ 23,50` (minimum w puli to dokładnie 23,50). Wady mierzone tu
są więc ortogonalne do progu: score liczono na tekstach angielskich przed
tłumaczeniem (klasa „zła polszczyzna" jest dla niego niewidzialna), a reranker
mierzy relewancję, nie odpowiadalność (pasaż o koniczynie pospolitej jest
relewantny dla pytania o koniczynę biskupią). Podnoszenie progu nie zastąpi
filtra: mediana score przy wadzie 25,62 wobec 27,25 bez wady — rozkłady mocno
się nakładają.

Istniejący sygnał mechaniczny słabo to łapie: mediana
`query_language_confidence` to 0,300 przy złej polszczyźnie wobec 0,400 przy
dobrej — rozkłady mocno się nakładają, więc próg na tej cesze nie zastąpi
pomiaru LLM.

## 2. Interpretacja

Model startowy SFT uczył się na puli, w której **~15% zapytań nie ma
odpowiedzi w swoim pasażu** — czyli wprost trenował generowanie zapytań
nieugruntowanych, dokładnie tej wady, którą potem łapaliśmy u niego w Task 06
(13,5% `chosen` par v3 odrzucone przez answerability). Te dwie liczby są
spójne i wskazują wspólne źródło: dane, nie architekturę.

## 3. Co z tego może wynikać (decyzje właściciela, nie tego raportu)

1. **Filtr puli i retrening SFT** — usunięcie ~25% wadliwych par i nowy
   adapter startowy. Koszt: nowy punkt startowy unieważnia porównywalność
   wszystkich dotychczasowych ramion Task 07 i wymaga powtórki łańcucha
   preferencji. Zysk potencjalnie największy, bo naprawia źródło.
2. **Filtr tylko przyszłych kohort** — nowe kohorty par liczą się z pulą
   przefiltrowaną, stare wyniki zostają nietknięte.
3. **Nic teraz** — wynik czeka do decyzji o finalistach.

Pomiar na próbce 12 000 z zamrożonym seedem wystarcza do decyzji (błąd
standardowy udziału ~0,4 pp); pełne 384 576 par to ~7 h serwera, potrzebne
dopiero, gdy zapadnie decyzja o filtrze.
