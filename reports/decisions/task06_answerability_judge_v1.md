# ADR V2-01: sędzia odpowiadalności — przypięcie wag i progi kalibracji

Status: **prospektywny, zamrożony przed wyprodukowaniem jakiegokolwiek werdyktu
kalibracyjnego**. Kontrakt: `task06-answerability-judge-v1`.
`human_evidence_claimed=false`, `used_for_pair_building=false`,
`task07_training_authorized=false`, `final_tests_used=[]`.

Data zamrożenia: 2026-08-19. Ten ADR jest commitowany **przed** uruchomieniem
sędziego na jakimkolwiek itemie kalibracyjnym; progów z §5 nie wolno zmieniać po
odczycie wyników.

## 1. Po co ten sędzia i dlaczego teraz

Audyt dual-LLM (pełny, 500 par) zmierzył lukę, której polityka par v1 nie
zamyka: **16,6% (`gpt-oss`) i 18,8% (`qwen`) stron `chosen` jest uznane za
nieodpowiadalne z pasażu**, mimo spełnionego round-tripu w top-20, a round-trip
odpowiadalności praktycznie nie różnicuje. Proxy odpowiadalności v1 — tania próba
zamknięcia tej luki cechami już policzonego scoringu — **oblało** swoje zamrożone
kryterium (czystość 0,8707 wobec progu 0,88;
[`task06_answerability_proxy_v1_2026-08-17.md`](../measurements/task06_answerability_proxy_v1_2026-08-17.md)),
i jego ADR zapisał, że lukę zamknie dopiero przypięty sędzia lokalny — tym ADR-em.

## 2. Przypięte wagi i odstępstwo od specyfikacji (jawnie)

Specyfikacja V2-01 wskazywała `qwen3.6-27b` Q4 przez ollamę na maszynie z 16 GB
VRAM. **Odstępujemy od tego i zapisujemy dlaczego:**

| | specyfikacja | ten ADR |
|---|---|---|
| model | `qwen3.6-27b` | **`Qwen3.8-27B FP8`** |
| kwantyzacja | Q4_K_M (~16,5 GB) | **FP8** |
| backend | ollama, lokalnie | **vLLM**, endpoint zgodny z OpenAI, druga maszyna operatora |

Powody, w kolejności wagi:

1. **Maszyna bazowa nie może uczciwie serwować 27B.** Ma 8 GB VRAM i 16 GB RAM,
   czyli ~20 GB użytecznego budżetu razem z KV-cache i runtime. Q4_K_M (16,5 GB)
   mieściłby się tylko na styk, z mmapowaniem z dysku. Zmierzone wykonalne
   minimum na tym sprzęcie to Q3_K_S (12,4 GB): **działa** i jest szybkie
   (3,4 s/item, 1045 itemów/h, parsowalność 8/8 w smoke'u), ale Q3 to znacznie
   gorszy punkt jakościowy dla sędziego rozstrzygającego subtelne przypadki.
2. **FP8 na mocnej karcie jest jakościowo lepszy niż Q3_K_S na 8 GB** i szybszy,
   więc wybór dyktuje jakość sędziego, nie wygoda.
3. Nowszy model tej samej rodziny; nadal **nie jest** rodziną generatora
   (Bielik) ani teacherem ablacji, więc self-preference nie wchodzi w grę.

**Ślad po ścieżce lokalnej:** wagi `qwen3.6:27b-q3_K_S`
(digest `418bbc5c98e5a6e5db38fa825f2fd5b8b72a5ec0616ea99a4b6346352d74edd7`) są na
maszynie bazowej zarejestrowane i **nie wyprodukowały ani jednego werdyktu
kalibracyjnego** — wykonano nimi wyłącznie ~20 wywołań pomiaru przepustowości,
nigdy nie zapisanych do journala i nigdy nie porównanych z żadną etykietą. Gdyby
kiedykolwiek miały być użyte do kalibracji, wymaga to osobnego ADR-u.

**Słabsze przypięcie tożsamości — zapisane wprost.** vLLM raportuje nazwę modelu
i własne metadane z `/v1/models`, a **nie** digest treści wag, więc pin jest tu
deklaracją operatora plus metadanymi serwera, nie pinem kryptograficznym. Jest to
realne osłabienie względem ścieżki ollamy i tak je raportujemy. Co **jest**
przypięte kryptograficznie: zbiór itemów (SHA-256 pakietu po obu stronach) i
prompt (SHA-256 `74d3ee07757decbdf5655e1878c070f66bf05c90a60bf4dc5f56b1c520cfee84`).

## 3. Zamrożony protokół wywołania

- prompt: `task06-answerability-pl-v1`, bajtowo niezmieniony;
- `temperature = 0`, `seed = 20260817`, `max_tokens = 24`,
  `response_format = {"type": "json_object"}`;
- **thinking wyłączony** (`enable_thinking: false`). Zamrożony prompt wprost
  wymaga „bez toku rozumowania”. Planowane ramię kontrastowe z włączonym
  thinkingiem **zostało porzucone**: na maszynie bazowej sam pomiar jego kosztu
  nie domknął się w rozsądnym czasie, a wyprowadzanie ramienia na drugiej
  maszynie wymagałoby własnych progów i podwoiłoby powierzchnię decyzyjną.
  Thinking pozostaje otwartą opcją dla osobnego, przyszłego ADR-u — nie wchodzi
  do tej kalibracji ani jako ramię, ani jako fallback;
- werdykt ∈ {`yes`, `no`, `uncertain`}; `uncertain` **blokuje rolę `chosen`, ale
  nie jest defektem** (nie wolno nim produkować rejected osi A);
- transfer: pakiet **label-free** (item_id, zapytanie, pasaż; pasaże
  zdeduplikowane), etykiety zostają na maszynie bazowej. Zdalny sędzia nie widzi
  żadnej etykiety, więc nie ma czego dostroić pod wynik;
- journal: jedna linia na werdykt, `flush` + `fsync`, wznawialny; import odrzuca
  itemy spoza pakietu, niezgodny prompt, mieszanie modeli i sprzeczne werdykty.

## 4. Zbiór kalibracyjny i co było znane przed zamrożeniem

Dwa źródła, oba już zebrane:

1. **Strony audytu Groq** — 1000 stron (500 par × 2), wszystkie z dwiema ocenami.
   Konsensus sędziów: **817 stron** (641 `yes`, 176 `no`).
2. **Klasy konstrukcyjne korpusu walidacyjnego nagrody** — 540 itemów:
   `good_specific` 180 (oczekiwane `yes`), `good_alternative` 180 (`yes`),
   `ungrounded` 180 (`no`). Etykiety pochodzą **z konstrukcji**, nie z sędziego.

Razem pakiet ma **1540 itemów** i **665 unikalnych pasaży**
(`items_sha256 = 31ac2436c34ef55d…`,
`item_ids_fingerprint = 655bf767a3d183dd…`).

Znane przed zamrożeniem (wyłącznie własności etykiet, **żadnego** werdyktu
sędziego): sufit szumu = zgodność sędziów Groq co do odpowiadalności **0,817**;
baza klasy większościowej na stronach konsensusowych **0,7846**; rozkład per rola
(`chosen` 373 `yes` / 50 `no`, `rejected` 268 `yes` / 126 `no`).

**Nie ma tu podziału fit/holdout i to jest celowe.** W przeciwieństwie do proxy,
które przeszukiwało 14 920 reguł i dlatego wymagało holdoutu, sędzia **nie ma
dopasowywanego parametru** — nie ma kanału przeuczenia, więc cały zbiór etykiet
jest jednym, jednorazowym odczytem.

## 5. Kryteria akceptacji (zamrożone przed odczytem)

Sędzia jest przyjęty jako sygnał odpowiadalności osi A wtedy i tylko wtedy, gdy
**wszystkie trzy** warunki są spełnione:

- **(K1) zgodność z konsensusem Groq**: `accuracy ≥ 0,85` **oraz**
  `balanced_accuracy ≥ 0,75` na 817 stronach konsensusowych.
  Sama accuracy nie wystarcza, bo reguła stała „yes” daje 0,7846; balanced
  accuracy 0,75 wymusza faktyczne wykrywanie klasy `no` (reguła stała daje 0,50).
  Dla kalibracji: oblane proxy miało na swoim holdoucie balanced accuracy 0,710,
  więc 0,75 jest realnym krokiem wyżej, a nie progiem osiąganym z marszu.
- **(K2) sanity na klasach z konstrukcji**: `ungrounded` → werdykt `no` w
  **≥ 0,80** itemów tej klasy, przy jednoczesnym **≤ 0,20** werdyktów `no` w
  `good_specific` i **≤ 0,20** w `good_alternative`. Warunek jest dwustronny,
  żeby nie nagradzać sędziego, który po prostu wszystko odrzuca. `uncertain`
  liczy się w mianowniku i **nie** liczy się jako trafienie (kryterium
  konserwatywne).
- **(K3) dyscyplina abstencji**: udział `uncertain` w całym pakiecie **≤ 0,25**.
  Sędzia abstynujący na co czwartym itemie nie może niczego bramkować.

Raportowane obowiązkowo obok kryteriów (nie są kryteriami): zgodność z każdym
sędzią Groq osobno z 95% CI, macierz pomyłek, wynik per rola (`chosen` /
`rejected`), rozkład werdyktów, udział `uncertain` per źródło, przepustowość.

**Przegląd ręczny (zobowiązanie wiążące).** Przy tak silnej kwantyzacji nie wolno
poprzestać na liczbach: przeglądam co najmniej **40 werdyktów** stratyfikowanych
po werdykcie, roli i po zgodności/niezgodności z konsensusem, i opisuję je w
raporcie. Zastrzeżenie chroniące przed dostrajaniem: przegląd może kalibrację
wyłącznie **unieważnić**, nigdy uratować — jeśli liczby przechodzą, a przegląd
pokazuje systematyczny błąd, sędzia jest odrzucony; jeśli liczby nie przechodzą,
przegląd nie może tego odwrócić.

## 6. Konsekwencje niedowiezienia (zamrożone)

Jeśli którykolwiek z K1–K3 nie przejdzie:

1. raportujemy to wprost, z liczbami, jako wynik negatywny;
2. sędzia **nie jest** używany w polityce par; oś A powstaje bez kontroli
   odpowiadalności, dokładnie jak po porażce proxy;
3. ADR V2-03 **nie może** przewidywać poprawy odpowiadalności `chosen` — luka
   pozostaje nazwanym długiem, a kolejną próbą byłby mocniejszy sędzia albo
   inna rodzina modeli, za osobnym ADR-em;
4. **nie wolno** ratować wyniku ani zmianą promptu, ani włączeniem thinkingu, ani
   zawężeniem podzbioru — każde z tych działań to nowa, prerejestrowana wersja
   kalibracji, raportowana obok tej.

Przy przejściu K1–K3 sędzia certyfikuje **wyłącznie kandydatów faktycznie
wchodzących do par** (`chosen` + rejected osi A), nie całe kohorty, a jego
werdykty nigdy nie porządkują kandydatów — filtrują.

## 7. Czego ten ADR nie zmienia

Nie zmienia `format.py`, bramki różnorodności, polityki par v1/v1.1 i jej
artefaktów, kontraktu audytu Groq, rubryki sędziów audytu, progu
`source_en_score ≥ 23,50`, splitów ani statusu Tasków 07/08. Nie buduje żadnej
pary. Nie dotyka `artifacts/task06/teacher_claude_v1/`. Nie otwiera testów
finalnych.
