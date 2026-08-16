# Task 06 — ablacja teachera na modelu API (Claude), ADR v1, 2026-08-14

## Kontekst

`tasks/06_candidate_scoring_and_preference_data.md` (sekcja „Bramka
różnorodności same-prompt”, wiersze 387–392) dopuszcza prospektywną **ablację
teachera**: większy, inference-only model generuje kandydatów na *dokładnie te
same prompty* co lokalny generator, jako dodatkowe źródło `chosen`, z osobnym
provenance, z zakazem sędziowania własnych kandydatów i z budżetem w skali kilku
tysięcy promptów na dobę. Rozszerzenie specyfikacji z 2026-08-13 wskazało jako
teachera lokalny `Qwen3.6-27B` Q4.

Właściciel udostępnił w oknie 2026-08-14 osobny, nietypowy budżet: tokeny modelu
asystującego (`claude-opus-5[1m]`), przy jednoczesnym pełnym obłożeniu GPU
kolejką bezobsługową (`configs/unattended_queue_2026-08-14.tsv`). Komenda
właściciela: generować dane bezpośrednio tokenami modelu asystującego, wybór
zakresu „T-C2, potem T-C1”.

## Decyzja

Powstaje kohorta `teacher_claude_v1`: **600 pasaży × 4 kontrolki × 4 kandydatów
= 9600 zapytań** napisanych bezpośrednio przez model asystujący, zapisanych
w `artifacts/task06/teacher_claude_v1/`. Jest to **jawna ablacja teachera**,
osobne ramię o osobnym provenance — nie jest to główna ścieżka danych Task 06
i nie zastępuje procedury W06+D01+selector zaakceptowanej przez właściciela.

### Pasaże i rozłączność

- źródło: `artifacts/task06/same_prompt_expansion_v3/cohort.records.jsonl`,
  pierwsze 600 klastrów w porządku `sha256(cluster_id)` rosnąco, split `train`;
- kohorta v3 była zamrożona quality-blind ADR
  [`task06_unattended_compute_window_2026-08-14.md`](task06_unattended_compute_window_2026-08-14.md)
  z zerowym nakładaniem na klastry smoke/pilot/v2 i z wykluczeniem 49352
  klastrów wspólnej selekcji 50k SFT; teacher dziedziczy te wykluczenia;
- materializacja: `scripts/slim_task06_cohort_passages.py` →
  `artifacts/task06/teacher_claude_v1/passages.slim.jsonl`
  (`record_count=600`, `natural_query_included=false`,
  `hard_negatives_included=false`, `quality_fields_included=[]`).

Naturalne `query` nie są przekazywane teacherowi. To jest istotne: teacher
widzi dokładnie to, co lokalny generator (sam pasaż), więc nie istnieje ścieżka,
w której teacher parafrazuje gold query zamiast rozwiązywać zadanie doc2query.

### Prompt: ten sam kontrakt, wszystkie cztery kontrolki

Kontrakt DPO wymaga wspólnego promptu. Lokalny run v3 przypisuje **jedną**
kontrolkę na pasaż (`sha256_balanced_round_robin_after_quality_blind_order`),
a przypisania nie da się odtworzyć bez uruchomienia jego kodu. Dlatego teacher
generuje kandydatów dla **wszystkich czterech** kontrolek D01 z zamrożonego
designu:

| kontrolka | form | intent | focus_bucket |
|---|---|---|---|
| c0 | `full_question` | `fact_lookup` | `beginning` |
| c1 | `keyword_query` | `definition` | `middle` |
| c2 | `full_question` | `procedure` | `end` |
| c3 | `keyword_query` | `entity_lookup` | `middle` |

Pozostałe pola `QueryControl` jak w lokalnym runie: `focus_mode=bucket`,
`length=medium`, `intent_applicable` nieustalone. Instrukcja przekazana
podsesjom jest semantycznie tym samym zadaniem co
`render_controlled_prompt` (`src/doc2query/models/templates.py`): jedno polskie
zapytanie odpowiadalne wyłącznie z pasażu, bez kopiowania długich fragmentów,
z zachowaniem nazw własnych/liczb/terminów, bez komentarza i numeracji.

Skutek: dla każdego pasażu v3 istnieje kandydat teachera na **identyczny**
prompt, jaki dostał lokalny generator, niezależnie od tego, którą kontrolkę
przypisał mu round-robin. Pozostałe trzy kontrolki są zapisane i legalne, ale
poza parą same-prompt dla tego pasażu.

### Czym ta kohorta różni się od kohorty studenta (istotne ograniczenie)

Teacher nie próbkuje rozkładu: cztery kandydatury na kontrolkę są **czterema
świadomie różnymi próbami napisanymi w jednym przebiegu**, a nie próbkami i.i.d.
z temperatury i seeda. Dlatego:

- `decoding`, `temperature`, `top_p`, `seed`, `token_logprobs` i
  `sequence_logprob` są dla teachera **niedostępne** i zapisane jako
  `not_applicable_api_teacher`;
- grupa teachera **nie wchodzi** do bramki różnorodności same-prompt: bramka
  mierzy kolaps rozkładu samplingu, a nie zbiór pisany z intencją bycia różnym.
  Stosowanie jej progów do teachera byłoby nadinterpretacją;
- teacher **nie zastępuje** żadnej grupy studenta i nie uzupełnia deficytu par
  z bramki różnorodności (ten deficyt domykają kohorty v2–v5).

### Provenance i nieprzypięte wagi

Największa słabość tej ablacji: teacher **nie ma przypiętych wag**. Zapisujemy
`author_model=claude-opus-5[1m]`, `author_session_date=2026-08-14`,
`pinned_weights=false`, `transport=api`. Wersja modelu może się zmienić między
sesjami, więc kohorta jest **nieodtwarzalna w sensie bit-exact** i nie może
pełnić roli zamrożonego komponentu kontraktu. Preferencją programu pozostaje
lokalny, przypięty `Qwen3.6-27B` Q4 (`tasks/06...md` wiersz 433); ta kohorta jest
tanim, wcześniejszym sygnałem, nie jego zamiennikiem.

### Granice

- Kandydaci teachera mogą wejść wyłącznie jako **dodatkowe źródło `chosen`**,
  z jawnym polem źródła, i wyłącznie po osobnym ADR zamrażającym politykę
  `chosen/rejected` (`dpo_pair_selector` ma nadal status
  `not_frozen_not_authorized`). Ten ADR par nie buduje i nie autoryzuje.
- Par zawierających kandydata teachera **nie ocenia** sędzia tożsamy
  z teacherem. Model asystujący nie jest sędzią w audycie dual-LLM tych par;
  sędziami pozostają `gpt-oss-120b` i `qwen3.6-27b`.
- Jeśli kandydaci teachera trafią kiedyś do `chosen`, DPO staje się w tej części
  **destylacją z modelu API**, a nie destylacją procedury W06+D01+selector
  opisanej w Task 07. Ramiona muszą być raportowane rozdzielnie; zmieszanie ich
  czyni wynik nieinterpretowalnym.
- Kohorta **nie wchodzi** do frozen train, nie zmienia progu
  `source_en_score >= 23.50`, nie dotyka splitów ani żadnego zamrożonego
  artefaktu. Zapis wyłącznie do nowego katalogu
  `artifacts/task06/teacher_claude_v1/`.
- Trwające runy v3/v4/v5 nie są modyfikowane ani wstrzymywane; teacher nie
  korzysta z GPU.
- `final_tests_used=[]`, `task07_training_authorized=false`,
  `task09_authorized=false`.

## Kryteria oceny (prerejestrowane, mierzone po zwolnieniu GPU)

Ablacja ma sens tylko wtedy, gdy da się powiedzieć, czy teacher jest lepszy od
studenta na tym samym promptcie. Po zwolnieniu GPU kohortę teachera scoruje
**ten sam** zamrożony kontrakt co studenta (primary builder, shadow veto-only,
corpus round-trip, komponenty leksykalne/focus/style/format):

- odsetek pasaży, w których najlepszy kandydat teachera ma wyższy primary score
  niż najlepszy kandydat studenta na tej samej kontrolce;
- to samo według shadow (kontrola niezależna) i corpus round-trip;
- rozkład `copy_density` i `format_valid` teachera vs studenta;
- odsetek przypadków, w których primary i shadow są niezgodne co do kierunku —
  raportowany, nie ukrywany.

Wynik nie awansuje niczego automatycznie: jest przesłanką do ADR polityki par.

## Kolejność wykonania

1. **CPU, wykonane:** 600 chudych pasaży + plan 24 shardów po 25 pasaży.
2. **Tokeny, to okno:** generacja 24 shardów × 25 pasaży × 4 kontrolki × 4
   kandydatów = 9600 rekordów, każdy shard do własnego pliku
   `shards/shard_NNN.jsonl`.
3. **CPU:** walidacja schematu, dedup w obrębie kontrolki, scalenie do
   `candidates.jsonl` z manifestem i hashami.
4. **GPU, po kolejce, nieautoryzowane tym ADR:** scoring zamrożonym kontraktem
   i pomiar kryteriów powyżej.
5. **CPU, osobny ADR, nieautoryzowane:** polityka `chosen/rejected` i budowa par.
