# Handoff sesji: polityka par v2, stan na 2026-08-20 wieczór

Dokument przekazania między sesjami. **Nie jest ADR-em i niczego nie zamraża.** Opisuje
stan faktyczny, wskazuje jedno następne zadanie i powtarza ograniczenia, które obowiązują
niezależnie od sesji. Wszystkie liczby poniżej są zmierzone i mają artefakty; nic tu nie
jest oczekiwaniem ani planem.

## 0. Pierwsza rzecz do zrobienia w nowej sesji

**`git push origin master`.** Siedem commitów jest zacommitowanych lokalnie i
**niewypchniętych**, bo GitHub był z tej maszyny nieosiągalny (connect timeout, kilka prób
w ciągu dnia):

```
e600166 Zmierz podaż osi A po certyfikacji: 2253 pary z kohort autoryzowanych
0541fdf Przyjmij sędziego odpowiadalności (K1-K3) i odrzuć paczkowanie (A/B)
a983adf Napraw budżet wyjścia paczki: 12N+20 było policzone dla zwięzłego JSON-a
b06ccc2 Pokaż sufit paczkowania przy starcie runu
9b034ee Dodaj paczkowanie zapytań jednego pasażu za bramką A/B
da1c9ab Dodaj zrzut faktycznie wysyłanych promptów sędziego do diagnostyki cache'u
ff50964 Domknij wartość werdyktu schematem (amendment protokołu V2-01)
```

Drzewo robocze jest czyste, walidacja przechodzi (ruff, `mypy src`, **700 testów**).

## 1. Co jest zamknięte i zmierzone

**Audyt dual-LLM v1 — KOMPLETNY** (500/500 par, `status: complete`). Baseline predykcji
dla v2: zgodność automatu z sędziami 0,7179 i 0,7080 wobec **0,8793 między sędziami** (CI
nie nachodzą); bramka fail-closed wyklucza **378/500 par (75,6%)**, czyli akceptowalnych
jest **122** — o rząd wielkości poniżej progu 1000; remisy 51,8% + 9,2% `both_bad` oraz
32,2%, **częstsze** przy wysokiej pewności; nieodpowiadalne `chosen` **16,6% / 18,8%**;
round-trip nie różnicuje odpowiadalności; `qwen` płaski w pasmach marginesu. Slice'y
konsensusu: im bardziej defektowy `rejected`, tym wyższa zgodność (0,909 dla braku
round-tripu vs 0,797 dla samego marginesu).
Raport: `reports/measurements/task06_dual_llm_pair_audit_2026-08-17.md`, sekcja „WYNIK PEŁNY".

**Proxy odpowiadalności v1 — ODRZUCONE.** Czystość na holdoucie 0,8707 wobec zamrożonego
progu 0,88. Reguła `longest_copied_ngram ≤ 3 AND pool_positive_score ≥ 7,777`. Zapisane
jako wynik negatywny; ADR i raport zostają jako ślad.

**Sędzia odpowiadalności V2-01 — PRZYJĘTY** (`Qwen/Qwen3.8-27B-FP8`, vLLM 0.27.1 na
maszynie operatora, prompt `task06-answerability-pl-v1`, dekodowanie `json_schema_enum`):
K1 accuracy **0,8566** (próg 0,85, n=809) i balanced **0,8878** (próg 0,75), K2
`ungrounded` → `no` **180/180** przy 0,1056/0,1222 odrzuceń w klasach dobrych (cap 0,20),
K3 abstencja **0,0065** (cap 0,25). Zapas K1 mały (0,66 pp), ale przejście stabilne —
drugi niezależny run dał 0,8557. **Powtarzalność serwera 0,9909** (14 różnic na 1540 mimo
`temperature=0` i seeda) — continuous batching jest niedeterministyczny. Przegląd ręczny
34 werdyktów wykonany; nie unieważnił kalibracji i wskazał, że referencja jest słabszą
stroną. Kierunek obciążenia: `recall_no` 0,9429 vs `recall_yes` 0,8328 → filtr osi A jest
**konserwatywny**.

**Paczkowanie zapytań — ODRZUCONE bramką A/B.** Zgodność 0,9052 wobec progu 0,98 oraz
istotny **jednostronny dryf w `uncertain`** (`yes→uncertain` 21 vs 2, `no→uncertain` 19 vs
2, p ≈ 0,0001). Do tego **żadnego zysku wydajnościowego**: 16,3 it/s paczkami wobec 19,1
i 17,0 it/s pojedynczo — serwer jest ograniczony **dekodowaniem, nie prefillem**, co obala
też wariant hybrydowy z wieloma pasażami. `--batch-size` zostaje na 1.

**Certyfikacja puli — WYKONANA.** 172 295 werdyktów, **zero kandydatów bez werdyktu**,
zero `out_of_schema`. Journale w `artifacts/task06/answerability_verdicts/`
(`verdicts_pool_authorized_single.jsonl` 23 676, `verdicts_pool_rest_single.jsonl` 148 619,
`verdicts_calib_single.jsonl` 1540 + drugi run + wariant paczkowy jako ślad).

**Podaż osi A — ZMIERZONA.** **2 253 pary z kohort autoryzowanych** (62,3% z 3 619 grup;
v1 199, v2 270, v3 1 784), 13 736 z v4–v11, razem 15 989. Filtr sędziego zachowuje
**79,5%** czystych `chosen`. Naturalny `rejected` osi A w **96,9%** grup → konstruowane
(V2-04) zbędne. Bramka różnorodności odrzuciła 1 grupę na 2 254.
Raport: `reports/measurements/task06_axis_a_supply_after_certification_2026-08-20.md`.

**Baseline monotonii (oś D) — ZMIERZONY** na 224 000 zapytań: monotonia słów początkowych
jest **dyktowana kontrolką**, nie kolapsem modelu (`intent=procedure` → „jak" w 100%,
`definition` → „definicja" w 99,5–100%, dwa słowa zbierają połowę populacji). Kontrolka
`length` **nigdy nie została użyta** (wszędzie `medium`). Baseline set-level dla GRPO:
distinct-1/2 per grupa 0,470–0,477 / 0,659–0,668 w v3–v11, v1 niżej (0,326/0,455).

**V2-02 (`focus_v2`)** zaimplementowane, kryterium niedowiezione (abstencja spadła o ~1 pp;
wąskie gardło to scorer leksykalny). **Oś C wypada z pierwszego wydania** decyzją
właściciela.

## 2. Jedno następne zadanie: ADR V2-03 + budowa par v2

Wsad jest **kompletny**, nic już nie blokuje. ADR V2-03
(`reports/decisions/task06_defect_pair_policy_v2.md`) musi zamrozić **przed zbudowaniem
pierwszej pary**:

1. **osie A i B** (oś C poza wydaniem), z definicjami dokładnie takimi, jakie zmierzył
   pomiar podaży: `chosen` = czysty wg polityki **i** werdykt sędziego `yes`;
   `rejected` osi A = werdykt `no` **albo** brak round-tripu @100; `uncertain` blokuje
   `chosen` i **nie jest** defektem;
2. **cięcie osi B** — wybrać spośród zmierzonych kandydatów inwentarza V2-00 i uzasadnić;
   rekomendacja z inwentarza: p75 `content_jaccard ≥ 0,0857` (podaż 4 857 grup; p90 daje
   1 900);
3. **kwoty osi** w próbce audytowej 500 par (rekomendacja: 250/250 z fallbackiem do
   podaży) oraz deterministyczne przypisanie osi grupie po haszu, max 1 para na prompt;
4. **tie-break** (decyzja o wariancie DivPO: `chosen` najbardziej odrębny, `rejected`
   najbardziej typowy) — zdecydować i zamrozić;
5. **margines primary wyłącznie jako sanity** `pool_margin > 0` po stronie `chosen`,
   nigdy jako porządkowanie;
6. **PREDYKCJE** z baselinem z pełnego audytu v1 (§1). Uwaga na dwa ograniczenia, które
   już obowiązują z wcześniejszych ADR-ów: predykcji „nieodpowiadalne `chosen` ≤ 5%" nie
   wolno wyprowadzać z ambicji, tylko ze zmierzonej czystości filtra; a przy przyjętym
   sędzim (`recall_no` 0,9429) dopuszczalna jest predykcja poprawy odpowiadalności — w
   przeciwieństwie do scenariusza po porażce proxy.

Potem: moduł `pair_policy_v2` na rusztowaniu v1 (pinowanie SHA-256, atomowa publikacja,
odmowa nadpisania, `authorized_cohorts` = v1+v2+v3, `margin_used_for_ordering=false`,
`axis` per para, testy CPU), budowa par, deterministyczny stratyfikowany eksport 500
ślepych par do **nowego** katalogu `artifacts/task06/preference_audit_v3_defect_pairs/`,
raport liczb per oś i przyczyn odrzuceń, a na końcu audyt dual-LLM par v2 (kontrakt Groq
bez zmian, `pair_count=500`).

## 3. Fakty operacyjne, które trzeba znać

- **Endpoint vLLM operatora nie należy do repozytorium.** Skrypt bierze go z
  `--base-url` albo `JUDGE_BASE_URL`. Nie wpisywać go do żadnego pliku w repo.
- Maszyna bazowa: **8 GB VRAM + 16 GB RAM** — nie serwuje 27B uczciwie. Wagi
  `qwen3.6:27b-q3_K_S` są tam zarejestrowane w ollamie (digest `418bbc5c98e5…`) i
  **nie wyprodukowały żadnego werdyktu**; ollama wymagała upgrade'u do 0.32.14.
- Praca na maszynie zdalnej idzie w `~/work/doc2queryllm` (nie zaśmiecać `~`), przez
  archiwum `artifacts/task06/judge_bundle_v1.tar.gz` + `scripts/task06_judge_remote.py`.
- Pakiety label-free: `answerability_packet_v1` (1540), `answerability_pool_authorized_v1`
  (23 676), `answerability_pool_rest_v1` (148 619).
- Sieć na maszynie bazowej bywa zawodna: GitHub i CDN HuggingFace odpadały wielokrotnie,
  a host vLLM ma **tylko rekord AAAA** i martwą trasę IPv6 — dlatego skrypt zdalny wybiera
  rodzinę adresów jawnie (`--address-family`, domyślnie ipv4).

## 4. Ograniczenia obowiązujące bez zmian

Nie ruszać: `format.py`, bramki różnorodności, polityki v1/v1.1 i jej artefaktów,
kontraktu audytu Groq, rubryki sędziów audytu, progu `source_en_score ≥ 23,50`, splitów,
`artifacts/task06/teacher_claude_v1/`. `task07_training_authorized=false` — **żadnego
treningu**. `final_tests_used=[]` wszędzie. Żadnych cronów i wyłączania maszyny. Predykcje
i progi zamrażane **przed** odczytem wyników, nigdy dostrajane po fakcie. Każda zmiana
statusu = aktualizacja `tasks/README.md` i pliku zadania **w tym samym commicie**. Nowe
ADR-y i raporty dopisywać do whitelisty `.gitignore`. Walidacja przed commitem: ruff,
`mypy src`, pełny pytest. Commity po polsku na master + push, bez wzmianek o
asystencie/AI i bez `Co-Authored-By`.

## 5. Czego nie robić (wnioski negatywne, żeby ich nie powtarzać)

- **Nie wracać do paczkowania** zapytań ani do wariantu z wieloma pasażami w jednym
  requeście: bramka A/B odrzuciła pierwsze, a pomiar czasu obalił premisę obu.
- **Nie liczyć na prefix cache** tego serwera (hit rate ~2% przy 84,7% requestów z
  identycznym prefiksem u poprzednika w pasie; przyczyna nierozstrzygnięta, diagnostyka w
  `artifacts/task06/judge_cache_diagnostics/`).
- **Nie używać proxy leksykalnego** jako filtra odpowiadalności — oblało własną bramkę.
- **Nie traktować `pool_margin` jako sygnału jakości** — audyt v1 to wyklucza.
