# Task 06 — amendment polityki par: dopuszczenie kohorty v3 do próbki 500 par (2026-08-17)

Amendment do [`task06_tentative_pair_policy_v1.md`](task06_tentative_pair_policy_v1.md).
Decyzja właściciela z 2026-08-17.

## Powód

Budowa par z kohort v1+v2 dała **447 par** z 828 grup `eligible` (55,8% w v1,
52,6% w v2), czyli **53 pary poniżej** rozwojowej bramki 500 par wymaganej przez
`tasks/06_candidate_scoring_and_preference_data.md` i pinowanej w zamrożonym
`configs/preferences/task06_groq_preference_audit_v1.json` (`pair_count: 500`).
Uruchomiła się ścieżka niedoboru z punktu 3 sekcji „Kolejność wykonania”
oryginalnego ADR, która wymaga w tej sytuacji **osobnej decyzji właściciela** i
wprost zabrania poluzowania jakiegokolwiek progu.

Właściciel wybrał uzupełnienie próbki z kohorty v3 (2791 grup `eligible`),
a nie obniżenie bramki.

## Decyzja

1. Powstaje nowy plik polityki `configs/preferences/task06_tentative_pair_policy_v1_1.yaml`
   z `policy_id = task06-tentative-pair-policy-v1.1`. **Jedyną** różnicą wobec v1
   jest lista `authorized_cohorts` rozszerzona o `same_prompt_expansion_v3` oraz
   wskazanie tego amendmentu w polu `adr`. Wszystkie progi, definicje metryk,
   veto shadow, filtry round-tripu, guard wtrącenia, kontrakt copy-risk, rola
   focusu, lista wykluczonych sygnałów i definicja straty próbki audytowej są
   **niezmienione**.
2. Plik v1 pozostaje **nietknięty**, razem z artefaktami
   `artifacts/task06/same_prompt_expansion_v{1,2}/tentative_pairs/`, które pinują
   jego SHA-256. Są zachowane jako ślad audytowy.
3. Pary buduje się ponownie dla v1, v2 i v3 pod polityką v1.1 do katalogów
   `tentative_pairs_v1_1/`. Ponieważ polityka par jest deterministyczna i jej
   progi się nie zmieniły, **fingerprinty par v1 i v2 muszą pozostać identyczne**
   z tymi zbudowanymi pod v1. Ta równość jest sprawdzana i raportowana jako dowód,
   że amendment nie zmienił sposobu budowy pary — a jeśli nie zachodzi, budowa
   jest wstrzymana i amendment unieważniony.
4. Próbka audytowa jest losowana ponownie z **połączonej** populacji v1+v2+v3 tą
   samą zamrożoną procedurą (ziarno 20260816, strata kohorta × `requested_form` ×
   pasmo marginesu, alokacja metodą największych reszt). Przy populacji istotnie
   większej niż 500 sampler **faktycznie losuje**, zamiast brać całą populację;
   pary niewylosowane pozostają nieoglądanym zapasem.
5. Nowy eksport ląduje w `artifacts/task06/preference_audit_v2/`. Eksport
   `preference_audit_v1` (447 par) jest **superseded**: pozostaje na dysku jako
   ślad audytowy i **nie wolno** go użyć do zbierania ocen.
6. Zamrożony `configs/preferences/task06_groq_preference_audit_v1.json` pozostaje
   **bez żadnej zmiany** — po uzupełnieniu do 500 par jego `pair_count` zgadza się
   z eksportem, a strażnik w `load_llm_audit_config` nie jest ruszany.

## Czego ten amendment nie robi

- **nie poluzowuje żadnego progu** — ani polityki par, ani bramki różnorodności,
  ani kontraktu audytu;
- **nie otwiera** kohort v4–v11 ani reszty v3 poza wylosowaną próbką: budowa par
  dla nich nadal wymaga pozytywnego audytu dual-LLM. `authorized_cohorts` ma po
  tym amendmencie trzy pozycje, nie jedenaście;
- **nie autoryzuje** treningu DPO. `task07_training_authorized=false`;
- **nie zmienia** kolejności „najpierw audyt, potem masowa produkcja par”. Zmienia
  wyłącznie to, z ilu kohort wolno pobrać materiał na sam audyt;
- **nie otwiera** testów finalnych; `final_tests_used=[]`.

## Uwaga o odwróceniu kolejności

Oryginalny ADR mówił, że v3–v11 dostają pary „dopiero po pozytywnym audycie”.
Ten amendment świadomie robi wyjątek dla v3 i tylko w zakresie próbki audytowej.
Uzasadnienie jest wprost metodologiczne, nie wydajnościowe: bez tego wyjątku
audyt musiałby objąć **całą** dostępną populację 447 par, więc nie zostałaby ani
jedna para nieoglądana, a rozwojowa bramka 500 par i tak nie byłaby spełniona.
Losowanie 500 z ~1850 par jest ściślejsze niż audyt „wszystkiego, co jest”.

`final_tests_used=[]`. Etap jest w całości CPU poza samymi wywołaniami audytu.
