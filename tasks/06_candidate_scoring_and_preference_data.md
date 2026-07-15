# Task 06 — Generacja kandydatów, scoring i dane preferencyjne

## Status

`TODO`

## Cel

Zbudować wysokiej jakości pary `chosen/rejected` dla DPO, nie myląc preferencji dokumentów z preferencjami odpowiedzi generatora.

## Zależności

Taski 02, 04 i 05 oraz stabilny checkpoint SFT.

## Generacja kandydatów

Dla każdego wybranego passage wygeneruj 4–8 kandydatów przez kombinację:

- 2–4 stylów adekwatnych do passage;
- 2–3 focusów;
- temperatur `0.3, 0.7, 1.0`;
- co najmniej dwóch seedów;
- opcjonalnie baseline’u bez treningu.

Zapisz pełne logprobs, parametry generacji, kontrolki i checkpoint ID, jeżeli biblioteka to wspiera.

Nie generuj preferencji na testach.

## Kandydaci negatywni

Źródła rejected:

1. gorszy, ale poprawnie sformatowany kandydat tego samego SFT;
2. kandydat z wysokim overlapem/kopiowaniem;
3. kandydat z niskim grounding margin;
4. kandydat dotyczący niepożądanego focusu;
5. kandydat duplikujący inne query;
6. ostrożnie: query powiązane z hard-negatywnym dokumentem.

Ostatnia kategoria nie może dominować, bo zbyt łatwe rejected uczą tylko tematyczności.

## Composite score

Zapisuj osobno każdy komponent oraz total. Przykładowe pola:

```json
{
  "ground_score": 0.81,
  "negative_margin": 0.42,
  "overlap_reward": 0.65,
  "focus_accuracy": 1.0,
  "style_accuracy": 1.0,
  "format_score": 1.0,
  "copy_penalty": 0.1,
  "answerability_flag": true,
  "total_score": 2.97
}
```

Nie usuwaj składowych po zsumowaniu.

## Budowa par

Preferowana metoda:

- wybierz top candidate jako `chosen`;
- wybierz `rejected` z dolnej części, ale o poprawnym formacie i minimalnej relewancji;
- wymagaj minimalnego `score_margin`;
- nie paruj identycznych lub niemal identycznych query;
- zachowaj rozkład typów błędów rejected;
- ogranicz liczbę par z jednego passage;
- nie pozwalaj, aby ten sam tekst query był zawsze chosen lub zawsze rejected bez analizy.

Warianty:

- pairwise top-vs-bottom;
- top-vs-near-miss;
- kilka rejected na chosen;
- listwise dane zachowane do przyszłych metod, nawet jeśli DPO używa par.

## Kontrola jakości preferencji

Automatycznie odrzuć:

- brak wyraźnego marginesu;
- na oba query nie można odpowiedzieć z pasażu;
- oba query identyczne po normalizacji;
- chosen z invalid format;
- chosen skrajnie ogólne;
- konflikt między reranker margin a answerability checks;
- wysoce niepewny focus.

Ręczna walidacja:

- min. 500 par na etapie rozwoju;
- min. 1000 par przed finalnym DPO;
- ślepa kolejność;
- preferencja człowieka i przyczyna;
- zgodność automatycznego rankingu z człowiekiem;
- analiza według źródła rejected.

## Leakage i splity

Preference train/dev/test muszą dziedziczyć split passage. Żaden passage/near-duplicate z preference dev/test nie może wejść do preference train.

## Wymagane skrypty

- `scripts/generate_candidates.py`
- `scripts/score_candidates.py`
- `scripts/select_candidate_sets.py`
- `scripts/build_preferences.py`
- `scripts/export_preference_audit.py`
- `scripts/import_preference_audit.py`

## Artefakty

- `candidates/<run_id>/*.parquet`
- `preferences/<version>/train.parquet`
- `preferences/<version>/dev.parquet`
- `preferences/<version>/test.parquet`
- `preferences/<version>/manifest.json`
- raport jakości i rozkładów.

## Kryteria akceptacji

- format zgodny z TRL DPO: prompt/chosen/rejected;
- każdy rekord ma wszystkie składowe score i provenance;
- zgodność automatu z człowiekiem jest raportowana;
- rejected nie są wyłącznie nonsensowne;
- score margin i typ rejected są zbalansowane;
- preference test jest zamrożony;
- continued-SFT dataset z samymi chosen jest generowany jako obowiązkowa kontrola.
