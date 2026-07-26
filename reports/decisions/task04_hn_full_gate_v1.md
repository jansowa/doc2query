# ADR: pełna bramka recept hard negative v1

Status: **ACCEPTED — pomiar dev domknięty, dotychczasowa recepta utrzymana**

Data decyzji: 2026-07-26

Zakres: Task 04, zamrożony `dev_intrinsic_rank10`; bez treningu i bez testów
finalnych

## Decyzja

Pełna bramka HN0/HN0+filter/HN1/HN2/HN3 jest wykonana. Dla kolejnych
porównywalnych probe pozostaje wcześniejsza recepta `HN0+filter` z polityką
`drop` i przypiętą kalibracją `possible_false_negative`. Pomiar nie daje
podstawy do zastąpienia jej HN1, HN2 ani HN3 i nie wybiera generatora.

HN2 nie jest promowane: zarówno primary, jak i shadow wskazują łatwiejszą pulę.
HN3 nie jest promowane z innego powodu. Jego perfect primary retrieval wynika
z konstrukcji positive-aware veto wykonywanego przez ten sam primary, więc nie
jest niezależnym potwierdzeniem. Shadow daje kierunek przeciwny, a winner
disagreement wynosi `9.81%`. HN1 nie odróżnia się statystycznie od HN0+filter
według primary, zaś shadow również daje przeciwny kierunek i disagreement
`10.06%`. Żaden nowy miner nie ma zatem zgodnej primary/shadow podstawy do
promocji. Testy finalne pozostają zamknięte.

## Wynik

Runner ocenił 1 000 prospektywnie wybranych query; wszystkie pięć ramion miało
pełne 10 negatywów dla wspólnych 775 query. Common-legal drop rate wyniósł
`22.5%`, dlatego wynik dotyczy wspólnej legalnej kohorty i nie może być
ekstrapolowany na odrzucone query.

| Ramię | nDCG@10 | MRR | Recall@1 | Primary/shadow disagreement |
|---|---:|---:|---:|---:|
| HN0 | 0.981781 | 0.976415 | 0.963871 | 0.055484 |
| HN0+filter | 0.981781 | 0.976415 | 0.963871 | 0.055484 |
| HN1 BM25 | 0.984691 | 0.979703 | 0.967742 | 0.100645 |
| HN2 bi-encoder | 0.993515 | 0.991430 | 0.987097 | 0.038710 |
| HN3 union+filter | 1.000000 | 1.000000 | 1.000000 | 0.098065 |

Shadow MRR/nDCG@10 na tej samej kohorcie wynoszą odpowiednio:

| Ramię | Shadow MRR | Shadow nDCG@10 |
|---|---:|---:|
| HN0 | 0.954032 | 0.963827 |
| HN0+filter | 0.954032 | 0.963827 |
| HN1 BM25 | 0.923700 | 0.939959 |
| HN2 bi-encoder | 0.974121 | 0.979670 |
| HN3 union+filter | 0.937520 | 0.952558 |

Bootstrap został wykonany dla primary; wartości shadow są confirmatory i nie
mają osobnego CI. Dlatego kierunek shadow jest podstawą do odmowy promocji,
nie do deklaracji statystycznej przewagi lub straty.

Paired-query bootstrap względem HN0+filter:

- HN1: nDCG@10 `+0.002910`, 95% CI `[-0.004737, +0.010803]`; MRR
  `+0.003288`, CI `[-0.006755, +0.013656]` — brak rozdzielenia;
- HN2: nDCG@10 `+0.011735`, CI `[+0.005866, +0.018135]`; MRR
  `+0.015015`, CI `[+0.007416, +0.023445]` — istotnie łatwiejsza pula;
- HN3: nDCG@10 `+0.018219`, CI `[+0.011714, +0.025488]`; MRR
  `+0.023585`, CI `[+0.015286, +0.032723]` — istotnie łatwiejsza pula.

HN0 i HN0+filter są identyczne na wspólnej legalnej kohorcie. Nie dowodzi to,
że filtr niczego nie zmienia: warunek pełnych 10 negatywów we wszystkich
ramionach wykluczył 225 query. Wcześniejszy P-03 pozostaje właściwym dowodem
dla konserwatywnego wyboru filtra.

## Provenance i ograniczenia

- artefakt: `artifacts/task04/hn_full_gate_v1/summary.json`;
- kopia pomiarowa: `reports/measurements/task04_hn_full_gate_v1/summary.json`;
- artifact fingerprint:
  `bc02f475c5955f32c92612519d35a21804db9ad2884426b788c17afee0d660a9`;
- common legal query: `775/1000`;
- bootstrap: 10 000 resampli po query, seed 42;
- `final_tests_used=[]`, `training_runs=[]`, `p06_t_modified=false`.

HN2 korzysta z zamrożonego dense cache naturalnego probe'a P05-GOLD, z maską
ograniczającą exact scan do dokumentów train. Nie jest to niezależny,
pretrained auxiliary bi-encoder, więc wynik HN2 nie może służyć jako ogólny
ranking rodzin minerów. HN3 używa primary zarówno do positive-aware veto, jak
i głównej metryki; jego perfect primary score jest zatem konstrukcyjny.

Jest to bramka recepty negatywów na naturalnym dev, a nie wynik probe
embeddera. Task 09 nadal zależy od Task 05–07 w zakresie dopuszczonym przez
ich własne bramki.
