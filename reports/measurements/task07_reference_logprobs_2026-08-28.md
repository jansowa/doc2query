# Task 07 — plan DPO i precompute logprobów referencji (2026-08-28)

## Status

Wykonane: **plan model-free** (`task07-dpo-plan-v1`) i **precompute logprobów
referencji** (`task07-precomputed-reference-logprobs-v1`) dla kohorty treningowej
par v3. To **nie trening**: forward-only pod `no_grad`, bez optymalizatora, bez
zapisu adaptera. `task07_training_authorized=false` pozostaje bez zmian, a zdjęcie
Groq z roli bramki ([amendment](../decisions/task06_v3_groq_role_amendment_2026-08-27.md))
autoryzacją treningu nie jest. `final_tests_used=[]`.

## 1. Plan (model-free)

`configs/train/task07_dpo_plan_v3_bottom.yaml`, `plan_id`
`task07-dpo-plan-v3-bottom-s42`, `plan_fingerprint`
`b1ab25b7071624cf60fb6ea1facd99a0e1eb207598fd35aa591b5c82e2e55c42`.

Hiperparametry są przepisane z `tasks/07_dpo_training.md` §Konfiguracja startowa
(beta 0,1; LR 1,0e-5; loss sigmoid; batch 1 × grad-accum 16), nie wymyślone tutaj.
Pozostałe wartości są **policzone**:

| pole | wartość | skąd |
|---|---|---|
| `max_length` | 768 | max `prompt+chosen` = 547 → zero truncacji |
| `max_prompt_length` | 704 | max promptu = 540 → prompt nigdy nie ucięty |
| `target_optimizer_steps` | 154 | `ceil(2461 / 16)`, jedno przejście po kohorcie |
| `target_token_budget` | 1 087 057 | suma tokenów par DPO w kohorcie treningowej |
| `cohort_fingerprint` | `1e7e2445087ebae5…` | kolejność 2 461 `preference_id` |

`max_length` 768 wybrano wprost, mimo że p99 `prompt+chosen` to 335: obcięcie
promptu zmienia warunkowanie, więc logprob referencji policzony na uciętym
prompcie nie byłby logprobem tej pary. Pomiar potwierdził cel — `prompt_truncated_count`
= **0**. Koszt jest zerowy, bo sekwencje są dynamicznej długości; 768 jest górnym
limitem, nie paddingiem.

**Tożsamość stosu jest zmierzona, nie wpisana.** `scripts/build_task07_dpo_plan_config.py`
liczy `artifact_fingerprint` bazy z treści snapshotu HF (sha256 każdego pliku,
9,5 GB) i `adapter_fingerprint` z treści katalogu adaptera:

- baza `speakleash/Bielik-4.5B-v3.0-Instruct` @ `4b1220a9…` →
  `8de58e412bb4393f86ef9eafaf0aa96363d26490c22e1f65174f216799dbf84a`
- adapter `D01-4.5B-STYLE-50K-S42` →
  `da862dd31af9cb7ca817c85fba43ecc86e298f211870645fdc9f5034163d2df5`

Tożsamość tokenizatora jest **skopiowana** z zamrożonego manifestu długości
tokenów, bo kontrakt wymaga tam równości, a manifest jest wcześniejszy.
Referencja jest identyczna ze stosem startowym — tego wymaga `DPOPlanConfig` i to
jest właśnie powód odrzucenia `trl.DPOTrainer` (patrz Status Task 07).

## 2. Precompute

`artifacts/task07/handoff_v3_bottom/reference_logprobs/`, RTX 3060 Ti, baza NF4 +
adapter SFT, bf16.

| pomiar | wartość |
|---|---|
| pary policzone | **2 461 / 2 461** (kohorta treningowa) |
| wznowione z journala | 0 (run bez przerwania) |
| prompty ucięte | **0** |
| peak VRAM | **2,72 GiB** (probe przewidywał 2,78 przy 768) |
| czas | 935,4 s = **15,6 min**; 0,380 s/para |
| walidacja | `validate_reference_logprobs` → 2 461 rekordów przyjętych |

Czas na parę wyszedł niemal dwukrotnie niższy od probe'u (0,711 s/para przy 512 na
parach syntetycznych z pasażem 180 słów): prawdziwe prompty są krótsze niż
syntetyczne, a przy dynamicznej długości to widać wprost w czasie.

Skrypt `scripts/precompute_task07_reference_logprobs.py` jest fail-closed w tej
kolejności: walidacja datasetu i planu → **przeliczenie** fingerprintów bazy i
adaptera i wymóg równości z planem → dopiero potem GPU. Precompute na innym stosie
niż zamrożony w planie kończy się odmową, a nie cichym artefaktem. Wznawianie idzie
po journalu i odmawia datasetu w innej kolejności.

Sam plik logprobów nie jest w gicie (dane runtime); w repozytorium są manifest,
`run_summary.json` i plan, wszystkie z fingerprintami.

## 3. Ślepy spot-check właściciela (przygotowany, nie wykonany)

`scripts/task07_owner_spot_check.py export` wylosował zamrożonym seedem 20260827
próbkę **50 par** z pełnego rozkładu (train+dev) i zapisał arkusz, w którym strony
są losowo przypisane do A/B (24 razy chosen jako A, 26 razy jako B). Klucz leży w
osobnym pliku. Testy pilnują ślepości arkusza: nie zawiera etykiet stron,
identyfikatorów par ani śladu głosowania.

Tryb `score` liczy zgodność z selektorem i dokładny dwustronny przedział
Cloppera-Pearsona. **Nie ma progu** i nie zostanie dopisany po zobaczeniu wyniku:
amendment §2.3 przewiduje kontrolę operacyjną, a nie bramkę, i **nie wolno**
raportować tego jako panelu AGENTS.md §9.3.

## 4. Co zostaje przed treningiem

1. Autoryzacja właściciela — `task07_training_authorized=false` to osobna flaga.
2. Spot-check 50 par (arkusz gotowy, odpowiedzi puste).
3. Podłączenie `doc2query train dpo` (nadal stub) i orkiestracja
   config → precompute → trening → manifest runu; runtime i logprobów już są.
4. Objętość: 2 461 par treningowych wobec ablacji zakładających 20k/50k/100k —
   ograniczenie zapisane, nie rozwiązane.
