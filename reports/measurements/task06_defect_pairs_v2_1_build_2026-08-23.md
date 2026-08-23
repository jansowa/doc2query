# Pomiar: budowa par v2.1, ślepy eksport komórki bramkowej i start audytu (2026-08-23)

Polityka i predykcje zamrożone **przed** tą budową:
[`task06_defect_pair_policy_v2_1.md`](../decisions/task06_defect_pair_policy_v2_1.md).
Amendment liczebności próby:
[`task06_groq_audit_sample_size_amendment_2026-08-23.md`](../decisions/task06_groq_audit_sample_size_amendment_2026-08-23.md).

Ten raport zawiera **wyłącznie** liczby o podaży i o próbce. Nie zawiera żadnego
wyniku audytu: w chwili jego zapisania nie istnieje ani jedna odczytana ocena
sędziego dla par v2.1. Liczba par **nie jest** miarą jakości polityki.

## 1. Budowa par: 2 253 pary, jedna oś

| kohorta | grupy `eligible` | pary | `judge_unanswerable` | `weak_corpus_round_trip` |
|---|---|---|---|---|
| `same_prompt_expansion_v1` | 362 | 199 | 132 | 67 |
| `same_prompt_expansion_v2` | 466 | 270 | 213 | 57 |
| `same_prompt_expansion_v3` | 2 791 | 1 784 | 1 359 | 425 |
| **razem** | **3 619** | **2 253** | **1 704** | **549** |

**Niezależna kontrola zgodności definicji.** 2 253 to **dokładnie** ta liczba,
którą zmierzono jako podaż osi A po certyfikacji puli
([raport](task06_axis_a_supply_after_certification_2026-08-20.md)) — a tamten
pomiar powstał zupełnie inną ścieżką kodu, przed istnieniem polityki v2.1. Zgodność
co do pary potwierdza, że zdjęcie osi B przywróciło definicję `chosen` bajtowo do
tej, na której podaż mierzono. Dla porównania: v2.0 zbudowała na osi A 2 086 par,
bo 167 grup parowalnych na osi A zabrał hasz przypisania osi.

Wynik jest zgodny z §5.1 ADR, który zapowiadał „co najmniej 2 086" i **zabraniał**
zakładania konkretnej liczby.

## 2. Ślepy eksport komórki bramkowej

- **800 par** z populacji 2 253, `shortfall = 0`, `development_gate_met=true`,
  `powered_sample_delivered=true`;
- **12 strat** (`cohort_id × rejected_defect_label × requested_form`), alokacja
  proporcjonalna metodą największych reszt, ziarno **20260823**, porządek `pair_id`;
- rozkład pierwotnej etykiety defektu w próbce: `judge_unanswerable` **606**,
  `weak_corpus_round_trip` **194** (proporcjonalnie do populacji 1 704 / 549);
- kohorty w próbce: v1 70, v2 96, v3 634;
- `rejected_verdict`: `no` 606, `yes` 194 (te drugie to defekt round-tripu);
  `chosen_verdict` = `yes` w 800/800;
- orientacja **400/400**, **800/800** zobowiązań `sha256(sól ‖ pair_id ‖ orientacja)`
  zweryfikowanych po odtworzeniu z opublikowanej soli;
- ślepe rekordy mają **dokładnie pięć** dozwolonych pól, zero pól zdradzających
  rolę, zero niespójności orientacji wobec klucza odślepiającego;
- **800 unikalnych grup i 800 unikalnych klastrów pasaży** — żadna grupa i żaden
  klaster nie powtarza się w próbce;
- `audit_ids_fingerprint = 3f26eb81f171d3b246b30d3c86be0548213978c8ba634be0c893c616401dcff7`;
- katalog: `artifacts/task06/preference_audit_v4_defect_pairs_v2_1/` — **nowy**;
  eksporty v1, v2 i v3 pozostają nietknięte.

**Zapas nieoglądany: 1 453 pary** (64,5% populacji). Komórka kotwic złotych (300
par, niebramkowa) powstanie z par **rozłącznych** z tą próbką i zostanie zbudowana
po starcie audytu bramkowego; do tego służy `--exclude-export-dir`.

## 3. Start audytu: pierwsze okno uruchomione

Uruchomiono `run_task06_groq_preference_audit.py --execute` na configu
`task06_groq_preference_audit_v2_1.json` (różnice wobec v1: wyłącznie
`pair_count`, `status`, `owner_waiver`, `scope_note` — sprawdzane testem). Plan
runu: **800 requestów** (400 na model, `batch_size=2`), 1 600 ocen.

Po pierwszych minutach run jest zdrowy: odpowiedzi spływają, zero zdarzeń błędu,
zużycie ~954 tok/request u `gpt-oss` (zgodne z 941 zmierzonymi w audycie v2).
Zgodnie z kontraktem run zatrzyma się czysto jako `incomplete_quota_deferred` po
wyczerpaniu dziennego budżetu (oczekiwane ~390 par dziś) i jest wznawialny tą samą
komendą w kolejnym oknie; journal per model jest trwały.

**Żaden wynik nie został odczytany.** Analiza i bramka V2.1-05 policzą się dopiero
po `status: complete`, dokładnie regułą §4.4 ADR (Clopper–Pearson, werdykt
trójwartościowy, P1 jako guardrail).

`task07_training_authorized=false`, `final_tests_used=[]`.
