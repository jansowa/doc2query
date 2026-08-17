# Pomiar: teacher API vs student D01 4.5B na identycznym promptcie (2026-08-16)

ADR kryteriów (zapisany przed generacją i przed dostępem do GPU):
[`task06_claude_teacher_ablation_v1.md`](../decisions/task06_claude_teacher_ablation_v1.md).
Amendment GPU:
[`task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md`](../decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md).
Artefakty: `artifacts/task06/teacher_claude_v1/scoring/` (9600/9600),
`comparison_vs_student_v3.json`.

Oba ramiona oceniał **ten sam** zamrożony kontrakt (primary
`sdadas/polish-reranker-roberta-v3` jako builder, shadow `BAAI/bge-reranker-v2-m3`
jako niezależna kontrola, corpus round-trip na zamrożonym BM25). Porównanie
obejmuje **600/600** par (pasaż, kontrolka) o **identycznym `prompt_sha256`**;
zero grup odrzucono z powodu niezgodności promptu.

## Wynik kryteriów z ADR

Kryterium literalne z ADR: „najlepszy kandydat teachera vs najlepszy kandydat
studenta”. Student ma jednak **8** próbek na prompt, a teacher **4** kandydatury,
więc porównanie „best-of-N” jest mechanicznie przychylne studentowi. Raportuję
więc obie wersje: literalną oraz **równobudżetową** (student ograniczony do
czterech slotów decodingu; oba rozłączne podzbiory 0–3 i 4–7 dla kontroli).
Wariant równobudżetowy jest analizą **post-hoc**, jawnie oznaczoną.

| sygnał | best-of-8 (literalny) | best-of-4 (sloty 0–3) | best-of-4 (sloty 4–7) |
|---|---|---|---|
| primary: teacher lepszy | **0.347** | **0.417** | **0.415** |
| shadow: teacher lepszy | 0.477 | 0.547 | 0.567 |
| corpus round-trip @20: teacher lepszy | 0.097 (85.0% remisów) | 0.195 (76.2%) | 0.163 (79.0%) |
| średni najlepszy primary (teacher vs student) | 10.50 vs 11.45 | 10.50 vs 11.04 | 10.50 vs 11.07 |

Niezgodność kierunku primary vs shadow: **22.0%** grup (przy best-of-8) — dwa
zamrożone sędziowie rozstrzygają to porównanie przeciwnie w ponad co piątym
pasażu. Jest to wyraźnie więcej niż 9.81% disagreement zmierzone w bramce HN,
i sama ta liczba wystarcza, by nie traktować żadnego z powyższych odsetków jako
mocnego dowodu.

**Odpowiedź na pytanie ablacji: nie.** Teacher API nie bije lokalnego D01 4.5B na
zamrożonym sygnale budującym — ani w wersji literalnej (34.7%), ani
równobudżetowej (41.6% średnio z dwóch rozłącznych podzbiorów). Shadow daje
lekkie wskazanie na teachera (54.7–56.7%), ale przy 22% niezgodności kierunku to
nie jest podstawa do żadnego wniosku. Corpus round-trip nie różnicuje: 76–85%
grup to remisy.

## Dlaczego student wygrywa maksimum, mimo że przegrywa średnią

Per kandydat, nie per grupa:

| | `copy_density` | `normalized_lcs` | `word_length` | `pool_margin` |
|---|---|---|---|---|
| teacher (9600) | 0.3061 | 0.4053 | 5.92 | **4.19** |
| student (24000, te 600 pasaży) | 0.2869 | 0.3565 | 5.19 | **2.88** |

Teacher jest **bardziej równy**: jego średni margines primary jest o 45% wyższy.
Student wygrywa nie średnią, a **maksimum** — wysokotemperaturowe próbkowanie
produkuje pojedyncze bardzo wysoko oceniane wyniki. To klasyczny efekt best-of-N,
a nie dowód lepszej jakości modelu.

`format_valid`: teacher 1.0000, student 0.9998. Oba ramiona są formalnie czyste.

## Potwierdzenie znaleziska P8 na prawdziwej kohorcie

Pomiar P8 na korpusie konstruowanym pokazał, że sygnały sędziowskie nagradzają
kopiowanie. Ta kohorta potwierdza to **na danych z produkcyjnego pipeline'u**:
wybór kandydata po `pool_positive_score` systematycznie sięga po kandydatów
bardziej kopiujących pasaż.

| | `copy_density` wszystkich | `copy_density` argmax-primary | argmax łamiący `copy_risk` |
|---|---|---|---|
| teacher | 0.3061 | **0.3702** | 149/600 = **24.8%** |
| student | 0.2869 | **0.3494** | 173/600 = **28.8%** |

Czyli **co czwarty–co trzeci** kandydat, którego wybrałby czysty argmax po
sygnale primary, łamie zamrożony guard `copy_risk` (`copy_density > 0.6`,
`normalized_lcs > 0.8`, `minimum_query_words: 4`). Guard w polityce par nie jest
formalnością — jest tym, co powstrzymuje politykę od uczenia DPO kopiowania.
Nie zmieniam ani guarda, ani polityki: oba są zamrożone.

## Wnioski i granice

- Kohorta teachera **nie dostarcza przesłanki** do dopuszczenia go jako
  dodatkowego źródła `chosen`. Wynik jest negatywny i użyteczny: zdejmuje presję
  wprowadzania do pipeline'u teachera o nieprzypiętych wagach, a lokalna
  procedura W06+D01+selector nie wypada gorzej od znacznie większego modelu API
  na zamrożonym sygnale budującym.
- Nie oznacza to, że zapytania teachera są „gorsze” dla wyszukiwania. Zmierzono
  wyłącznie zgodność z sędziami i round-trip; jedyną wiążącą metryką programu
  pozostaje wynik probe-embeddera na naturalnych zamrożonych zapytaniach, którego
  tu **nie** uruchamiano.
- Teacher nie ma przypiętych wag (transport API), więc kohorta pozostaje
  nieodtwarzalna bit-exact i pozostaje osobnym ramieniem ablacyjnym.
- Kohorta teachera **nie weszła** do frozen train, do żadnej kohorty
  preferencyjnej, ani do par; niczego na niej nie trenowano. Budowa par z
  udziałem teachera i audyt dual-LLM pozostają nieautoryzowane.
- `final_tests_used=[]`.
