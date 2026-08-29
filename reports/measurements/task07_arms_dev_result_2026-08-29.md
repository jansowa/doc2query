# Task 07 — sześć ramion wytrenowanych, wynik na dev (2026-08-29)

## Status

Wykonane i zakończone: **trzy ramiona × dwa warianty par** (`bottom` i
`near_miss` — zarejestrowana ablacja z `tasks/07_dpo_training.md` §Ablacje),
wszystkie na zamrożonych planach, z autoryzacją
[`task07_training_authorization_2026-08-28.md`](../decisions/task07_training_authorization_2026-08-28.md).
Metryki poniżej to **dev** (269/232 par) — pomocnicze. **Żadne z tych liczb nie
są wynikiem programu**: kryterium rozstrzygającym jest probe embedder na
naturalnych zamrożonych zapytaniach (AGENTS.md §9.2), którego te runy jeszcze
nie mają. `final_tests_used=[]`.

Warianty różnią się wyłącznie stroną `rejected` (ten sam prompt, pasaż i
`chosen`); plany: `task07-dpo-plan-v3-bottom-s42` (`b1ab25b7…`, 154 kroki,
2 461 par) i `task07-dpo-plan-v3-nearmiss-s42` (`5515a56f…`, 138 kroków,
2 198 par). Wszystkie runy: seed 42, beta 0,1, LR 1e-5, sigmoid, max_length 768,
batch 1×16, checkpointy co 25 kroków (dwa runy przeszły restart maszyny i
wznowienie z checkpointu wraz ze stanem AdamW). Peak VRAM ≤3,92 GiB.

## 1. Wynik na dev

| wariant | ramię | NLL/token `chosen` | margin acc | implicit reward acc | kroki | czas |
|---|---|---|---|---|---|---|
| bottom | DPO | 0,6056 → **1,4709** | 0,9368 → 0,9851 | 0,9517 | 154 | 58 min |
| bottom | continued SFT | 0,6056 → **0,5325** | — | — | 154 | 10 min¹ |
| bottom | weighted SFT | 0,6056 → 0,5335 | — | — | 154 | 29 min |
| near_miss | DPO | 0,6084 → **1,3223** | 0,7696 → 0,9522 | 0,9043 | 138 | 51 min |
| near_miss | continued SFT | 0,6084 → **0,5235** | — | — | 138 | 25 min |
| near_miss | weighted SFT | 0,6084 → 0,5242 | — | — | 138 | 25 min |

¹ czas procesu po wznowieniu; ramię przeszło restart maszyny w kroku ~120.

Kontrole nie widzą strony `rejected`, więc margin/implicit reward nie są dla
nich zdefiniowane w treningu; porównanie preferencyjne wszystkich sześciu
adapterów wymaga osobnej ewaluacji tym samym pomiarem (`dev_metrics` na parach)
— zaplanowanej razem z probe.

## 2. Trzy obserwacje, wszystkie przewidziane przed wynikami

**2.1. DPO kupuje margines, płacąc prawdopodobieństwem `chosen` — w obu
wariantach.** NLL/token rośnie 2,2–2,4×, gdy margines rośnie. To znany tryb
straty sigmoid DPO (nagradza różnicę, nie poziom) i był zapisany w
[raporcie pierwszego ramienia](task07_reference_logprobs_2026-08-28.md) oraz
w diagnostyce kontrastu **przed** uruchomieniem pozostałych pięciu ramion.
Continued SFT robi odwrotnie (NLL spada). Czy któreś z tych zachowań daje
lepsze zapytania — orzeka wyłącznie probe.

**2.2. Pary near_miss są mierzalnie trudniejsze i domykają mniejszy zapas.**
Punkt startowy trafia 76,96% par near_miss wobec 93,68% bottom — zgodnie z
pomiarem pokrycia (`rejected` 0,333 vs 0,200). DPO na near_miss podnosi
trafność o **+18,3 pp** (0,770→0,952) wobec **+4,8 pp** na bottom
(0,937→0,985), przy *mniejszym* koszcie NLL (1,32 vs 1,47). Na dev wariant
near_miss uczy więc więcej za mniej — dokładnie to przewidywała diagnostyka
(§4.3: „wynik bottom vs near_miss staje się pomiarem informatywnym samym w
sobie").

**2.3. Wagi percentylowe nic nie zmieniają.** Weighted SFT ≈ continued SFT w
obu wariantach (ΔNLL ≤ 0,001). Spodziewane: wagi [0,5; 1,5] z rangi
`pool_margin` to za słaby gradient, by przy 138–154 krokach odróżnić ramiona.
Ramię pozostaje ważną kontrolą kontraktową, ale na dev jest nieodróżnialne.

## 3. Artefakty

| run | adapter (fingerprint) |
|---|---|
| `runs/T07-V3-DPO-S42` | `70060bd7a7a1…` |
| `runs/T07-V3-CSFT-S42` | `ee6a4aceeb8c…` |
| `runs/T07-V3-WSFT-S42` | `29707563ad38…` |
| `runs/T07-NM-DPO-S42` | `50869c633dae…` |
| `runs/T07-NM-CSFT-S42` | `9f49fe79b934…` |
| `runs/T07-NM-WSFT-S42` | `f3ce3d65c999…` |

Każdy run ma `run_manifest.json` (kontrakt `task07-dpo-run-v1`, self-fingerprint,
plan, kohorta, budżet, dev start/end) i pełną historię kroków w `history.jsonl`.

## 4. Co dalej (nierozpoczęte)

1. **Ewaluacja probe** wszystkich sześciu adapterów + punktu startowego na
   naturalnych zamrożonych zapytaniach — to jest właściwy wynik Task 07;
   wymaga generacji zapytań adapterami i przebiegu probe (~1500 s/run × pary
   seedów), czyli osobnego okna GPU.
2. Porównanie preferencyjne sześciu adapterów wspólnym `dev_metrics`.
3. Równolegle: prospektywny ADR pipeline'u par z wadami
   ([projekt](../plans/task07_defect_pair_pipeline_design_2026-08-29.md)) —
   trzecia kohorta o twardszym kontraście niż obie powyższe.
