# Task 05 — natural-query calibration i prospektywne audyty

## Stan

Pakiet CPU-only został zaimplementowany i uruchomiony niezależnie od D01.
Źródłem jest wyłącznie zamrożony `dev_intrinsic_rank10`; nie odczytano wyników
D01 ani testów finalnych. Kontrakt
`configs/evaluation/task05_natural_audits_v1.json` prospektywnie przypina seed
`20260726`, rozmiary 500/200, jednego właściciela-oceniającego, osie
stratyfikacji, obsługę małych domen, wersje
reguł i ekstraktora, jawne `unknown`/abstention, `intent_applicable` oraz
`final_tests_used=[]`. Nie definiuje arbitralnej bramki style accuracy.

Pełna materializacja objęła 6598 naturalnych query. Powstały dokładnie:

- 500 rekordów ślepego audytu `form`/`intent` i oddzielny machine key;
- 200 unikalnych pasaży ślepego audytu koncepcji i oddzielne propozycje;
- dwa formularze adjudykacji i instrukcja kodowania;
- per-record JSONL, opisowa kalibracja JSON/Markdown, identity, fingerprinty,
  journale i atomowy manifest końcowy.

Artefakty są lokalne w `artifacts/task05/natural_audits_v1/` i celowo nie są
commitowane, ponieważ zawierają tekst danych. Manifest ma status
`materialized_unreviewed`; audyty etykiet i koncepcji są `NOT MEASURED`.

Przygotowano również automatycznego anotatora Groq jako zastępstwo dla
ręcznego wypełniania formularzy. Dwa niezależnie limitowane workery używają
`qwen/qwen3.6-27b` (reasoning `none`) i `openai/gpt-oss-120b` (reasoning
`low`). Każdy request i response trafia do trwałego JSONL, a resume pomija
kompletne odpowiedzi. Plan v2 przenosi siedem poprawnych ocen z pilota i
obejmuje 693 pozostałe rekordy w 224 paczkach. LLM-y są automatycznym proxy,
nie ludzkimi oceniającymi; zgodność człowieka pozostanie `NOT MEASURED`.

## Wynik opisowej kalibracji (nie accuracy)

Na 6598 rekordach reguły przewidziały 4453 `full_question`, 1720
`keyword_query` i 425 `unknown` dla formy. Dla intencji: 3888 `fact_lookup`,
1165 `definition`, 500 `entity_lookup`, 597 `procedure`, 36 `comparison` i 412
`unknown`. Łączne abstention wynosi 425. `intent_applicable=true` wystąpiło
dla 5270 rekordów, a 1328 pozostało nierozstrzygniętych (`null`). Frozen dev
ma tylko jedno źródło/domenę `speakleash/msmarco_pl`; raport nie udaje więc
kalibracji między domenami. Te liczby opisują automatyczne predykcje, nie ich
poprawność.

Kluczowe fingerprinty materializacji:

- frozen cohort: `235d9b81e04ddc5e74bd2bbe884055dd74f03b6706e6030e88a4f918ac2ffab6`;
- identity: `80256116cd0cfb291a896e4f8dc756468f414bc823e608d1157a4c5588494e9d`;
- label blind CSV: `9dececc677350e5f66be0b633de411fb6c9d18ae7152dd6bbd47df806a8783b2`;
- label machine key: `f8fe3e03fb55502e7ce79545788f169b7fe9582874280875279729fb00ab1204`;
- concept blind CSV: `a6f46ecce4f706ec7284c0553d63772e4983002214714cffc8c65953bebfe273`;
- concept proposals: `b5729433ae88d0a95bf424e2d342d2056a91b395d0b7aba9c37d30dd785c862f`.

## Wznawialność i fail-closed

Kalibracja i ekstrakcja mają osobne trwałe journale JSONL. Czytnik naprawia
wyłącznie crash-truncated ostatnią linię, a ukończony prefiks nie jest liczony
ponownie. Identity obejmuje kontrakt, frozen manifest/cohort, kolejność ID,
seed i `final_tests_used=[]`. Drift odmawia wznowienia; jawna opcja
`--archive-incompatible` przenosi częściowy stan do odzyskiwalnego archiwum.
Finały JSONL oraz CSV są zapisywane atomowo. Progress pokazuje licznik,
remaining, throughput i ETA w sekundach.

Agregatory nie zwracają `complete`, dopóki każdy przypadek nie ma wymaganej
jednej oceny. Przy jednym oceniającym zgodność Cohen/Fleiss jest jawnie
`NOT MEASURED`; nie jest zastępowana zerem. Jeżeli później dojdą kolejne oceny,
wszystkie rozbieżności wymagają adjudykacji. Raport
etykiet zawiera confusion matrix, precision/recall/F1 per klasa, coverage,
accuracy na nie-abstention, wyniki per domena, reliability bins i Cohen/Fleiss
kappa. Raport koncepcji obejmuje correct/spurious/missing, liczby/jednostki,
fragmentację, duplikaty i przydatność coverage-aware wraz ze zgodnością.

## Komendy

Pełna materializacja (już wykonana; komenda jest bezpiecznie wznawialna):

```bash
.venv/bin/python scripts/task05_natural_audits.py materialize \
  --contract configs/evaluation/task05_natural_audits_v1.json \
  --output-dir artifacts/task05/natural_audits_v1
```

Plan Groq bez użycia limitu API:

```bash
.venv/bin/python scripts/run_task05_groq_audits.py --plan-only
```

Docelowe uruchomienie wykonuje właściciel projektu. Bieżący ledger ma jeden
request GPT-OSS przerwany po zapisie `request_started`, dlatego pierwsze
wznowienie wymaga jawnej zgody na jego potencjalne, jednorazowe powtórzenie:

```bash
.venv/bin/python scripts/run_task05_groq_audits.py \
  --allow-ambiguous-resend
```

Kolejne zwykłe wznowienia nie wymagają flagi, o ile poprzedni proces zakończył
się czysto. Klucz jest czytany z pola `api_key` w `.env` i nigdy nie trafia do
ledgerów. Dwa workery mają osobne liczniki, odstęp 2,1 s, lokalną rezerwację
TPM/dzień i retry wyłącznie dla jednoznacznego HTTP 429. Flagi
`--allow-ambiguous-resend` nie należy używać rutynowo: dopuszcza duplikat tylko
wtedy, gdy operator świadomie rozstrzyga request bez zapisanej odpowiedzi.

Po statusie `complete` wyniki automatyczne agreguje się tymi samymi,
fail-closed agregatorami, lecz do osobnych katalogów raportowych:

```bash
.venv/bin/python scripts/task05_natural_audits.py aggregate-labels \
  --machine-key artifacts/task05/natural_audits_v1/label_audit_machine_key.jsonl \
  --ratings artifacts/task05/groq_llm_audit_v2/label_llm_ratings.csv \
  --adjudication artifacts/task05/natural_audits_v1/label_adjudication.csv \
  --output-dir reports/measurements/task05_groq_label_audit_v2

.venv/bin/python scripts/task05_natural_audits.py aggregate-concepts \
  --machine-proposals artifacts/task05/natural_audits_v1/concept_audit_machine_proposals.jsonl \
  --ratings artifacts/task05/groq_llm_audit_v2/concept_llm_ratings.csv \
  --adjudication artifacts/task05/natural_audits_v1/concept_adjudication.csv \
  --output-dir reports/measurements/task05_groq_concept_audit_v2
```

Raporty te muszą pozostać oznaczone jako LLM-proxy; nie są pomiarem człowieka
ani agreement między oceniającymi.

Właściciel projektu wypełnia po jednej kopii `label_audit_blind.csv` i
`concept_audit_blind.csv`, nie otwierając wcześniej machine key/proposals.
Następnie uruchamia:

```bash
.venv/bin/python scripts/task05_natural_audits.py aggregate-labels \
  --machine-key artifacts/task05/natural_audits_v1/label_audit_machine_key.jsonl \
  --ratings path/to/labels_owner.csv \
  --adjudication artifacts/task05/natural_audits_v1/label_adjudication.csv \
  --output-dir reports/measurements/task05_natural_label_audit_v1

.venv/bin/python scripts/task05_natural_audits.py aggregate-concepts \
  --machine-proposals artifacts/task05/natural_audits_v1/concept_audit_machine_proposals.jsonl \
  --ratings path/to/concepts_owner.csv \
  --adjudication artifacts/task05/natural_audits_v1/concept_adjudication.csv \
  --output-dir reports/measurements/task05_concept_audit_v1
```

Do kolejnej sesji należy wrócić z jednym kompletnym plikiem ocen etykiet i
jednym kompletnym plikiem ocen koncepcji. Adjudykacja nie jest potrzebna bez
rozbieżności między osobami. Wtedy wolno raportować ręczne accuracy i wynik
audytu ekstrakcji, ale zgodność między oceniającymi pozostanie `NOT MEASURED`.
