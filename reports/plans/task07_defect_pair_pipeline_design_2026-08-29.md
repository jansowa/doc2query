# Projekt pipeline'u par preferencyjnych z wadami nazwanymi z konstrukcji (2026-08-29)

## Status

**Projekt do dyskusji, nie zamrożony ADR.** Masowa generacja wymaga prospektywnego
ADR spisanego przed budową jakiejkolwiek kohorty treningowej; ten dokument jest
jego szkicem popartym eksploracją 40 próbek na Groq
(`artifacts/task06/teacher_defect_explore_v1/full_samples.jsonl`, prompt
`task06-teacher-defect-pl-v1`, `qwen/qwen3.6-27b`, temperature 0). Autoryzacja
kierunku: decyzja właściciela z 2026-08-28 („możemy używać Qwen-a do generowania
par […] koniecznie trzeba w metadanych zapisywać, jakiej klasy problemu dotyczy
ta para"). `final_tests_used=[]`.

## 1. Problem, który to rozwiązuje (zmierzony, nie odczuty)

- Pary v3 mają kontrast tematyczny, nie jakościowy: w **75,3%** strona `rejected`
  jest nie o tym pasażu; obie strony ugruntowane ma **7,1%** par
  ([diagnostyka](../measurements/task07_pair_contrast_diagnostic_2026-08-28.md)).
- Punkt startowy trafia **93,68%** par dev bez treningu; właściciel w ślepym
  spot-checku 95,45% — zadanie w parach jest w dużej mierze umiane.
- Pierwszy run DPO na tych parach podniósł margines (0,937→0,985), płacąc
  2,4× spadkiem prawdopodobieństwa `chosen` (NLL/token 0,61→1,47).

Wniosek: wąskim gardłem są **negatywy**. Studenckie complety dla tego samego
promptu nie zawierają trudnych negatywów — słabe kandydaty Bielika są po prostu
nie na temat.

## 2. Podział organiczne / syntetyczne — decyzja projektowa z uzasadnieniem

| element pary | źródło | dlaczego |
|---|---|---|
| pasaż + prompt | **organiczne** — dokładnie te same 2 730 grup co pary v3 | zero nowych pytań o pulę pasaży; dziedziczy rozłączność klastrową, kontrakt same-prompt i kontrolki bez żadnej nowej decyzji |
| `chosen` | **organiczne** — zwycięzca turnieju v3 (student, potwierdzony 6/6) | teacher jako autor *lepszych* zapytań już przegrał pomiar (34,7%/41,6%, [raport](../measurements/task06_teacher_vs_student_v3_2026-08-16.md)); trenowany model ma zostać we własnym rozkładzie, a DPO i tak spycha logprob `chosen` — destylacja stylu pogorszyłaby oba problemy |
| `rejected` | **syntetyczne** — teacher wstrzykuje jedną nazwaną wadę do zapytania na temat pasażu | jedyny brakujący składnik; wada znana **z konstrukcji** daje etykietę weryfikowalną, jak w kalibracji selektora v3 (dowód mocniejszy niż „inny model się zgadza") |
| `rejected` — druga populacja | **organiczne** — dotychczasowe bottom/near-miss z etykietą `off_topic` | kotwica: bez niej DPO mogłoby się uczyć „unikaj stylu teachera" zamiast „unikaj wady"; mieszanka dwóch populacji negatywów utrudnia ten skrót |

Kluczowa własność: **nowe pary różnią się od par v3 wyłącznie stroną `rejected`.**
Prompt, pasaż i `chosen` są bajt w bajt te same, więc porównanie ramion mierzy
dokładnie wartość negatywów.

## 3. Taksonomia wad (metadane `defect_class`, wymóg właściciela)

Klasy celują w problemy, które program już nazwał (rubryki R1/R2, klasy złe
kalibracji v3, obserwacje spot-checku):

| klasa | wada | weryfikacja mechaniczna | weryfikacja LLM |
|---|---|---|---|
| `copy_phrasing` | kopiuje ciągły fragment ≥5 słów z pasażu | **deterministyczna** (LCS słów) | zbędna |
| `not_answerable` | encje/terminy z pasażu, ale pasaż nie zawiera odpowiedzi | częściowa (pokrycie encji) | **answerability judge v1** (istnieje, zamrożony) — musi orzec NIE; dla `chosen` TAK |
| `too_general` | pasuje do tysięcy pasaży, konkret usunięty | Jaccard vs `chosen` (łapie równoważność) | sędzia: „czy to zapytanie wyróżnia ten pasaż?" |
| `answer_leak` | zawiera odpowiedź, o którą pyta | częściowa (tokeny faktu obecne w zapytaniu) | sędzia potwierdza obecność faktu-odpowiedzi |
| `wrong_form` (opcjonalna) | treść dobra, złamany kontrakt Forma/Długość | **deterministyczna** (regex kontraktu) | zbędna; uwaga: kalibracja v3 wykazała, że tej klasy sędzia ślepo nie ocenia — tu etykieta jest z konstrukcji, więc problem znika |

## 4. Co pokazała eksploracja (40 próbek, przegląd ręczny)

Jakość generacji jest wystarczająca, ale **generacja bez weryfikacji nie jest**:

- `copy_phrasing` — 10/10 z wadą obecną; np. „miasto w hrabstwie San Mateo w
  Kalifornii, w Stanach Zjednoczonych" przy chosen „jak brzmi nazwa miasta
  Belmont w Kalifornii".
- `not_answerable` — najmocniejsza klasa, 10/10 wiarygodnych: „jakie są ceny
  usług SWI dla platform handlowych", „mediana dochodu w Sandpoint Idaho",
  „data premiery Forda C-Max Energi" — na temat, z encją, bez odpowiedzi.
- `too_general` — **3/9 wadliwych**: „co to jest swi" ≈ chosen „kim jest swi"
  (para prawie równoważna — trująca dla DPO); „gdzie leży Belmont" jest
  odpowiadalne z pasażu i nie gorsze od chosen. To dokładnie ten tryb awarii,
  który łapie próg Jaccard + sędzia dystynktywności.
- `answer_leak` — 8/10 czysto; 2/10 pomylenie klasy (np. „czy china anne mcclain
  jest singielką" to `not_answerable`, nie leak). Nie psuje pary (nadal gorsza),
  ale psuje etykietę — stąd weryfikacja klasy, nie tylko jakości.

Operacyjnie: Groq wymaga `User-Agent` (Cloudflare 1010) i `reasoning_effort:
"none"` (inaczej Qwen przepala limit tokenów na myślenie i `json_object` pada);
darmowy tier ma niskie RPM → backoff jest w skrypcie.

## 5. Pipeline warstwowy (serwer Qwen3.8-27B; wydajność jawnie nie jest celem)

```
S1 GENERACJA      teacher, per (grupa × klasa), temperature 0, JSON
S2 FILTRY CPU     deterministyczne: LCS dla copy_phrasing, Jaccard vs chosen
                  ≤ próg (anty-równoważność), regex kontraktu formy, granice
                  długości, dedup w grupie
S3 ANSWERABILITY  judge v1: chosen→TAK zawsze; not_answerable→NIE;
                  pozostałe klasy→TAK (wada nie może niszczyć tematu)
S4 PREFERENCJA    potwierdzenie chosen > rejected z pozycją zamienioną,
                  jednomyślnie (jak v3); do tego weryfikacja KLASY wady
S5 ANTY-SKRÓT     trywialny klasyfikator powierzchniowy (długość, interpunkcja,
                  pierwsze słowo) chosen-vs-rejected; wysoka separowalność
                  cechami stylu = czerwona flaga przed treningiem
S6 SKŁADANIE      metadane: defect_class, prompt_version, model teachera,
                  głosy weryfikacji, pomiary mechaniczne, wersje progów
```

Odrzuty z S2–S4 wolno regenerować kolejnymi przebiegami (właściciel dopuszcza
wiele przejść) — ale każda para w kohorcie musi mieć komplet zaliczonych warstw.
Wszystko wznawialne po journalu, jak turniej v3.

Budżet: 2 730 grup × 4–5 klas × (1 generacja + 2–3 weryfikacje) ≈ **30–40k
wywołań** — jedna–dwie noce na serwerze bez limitów; eksploracje promptów dalej
na Groq w ramach dziennych okien.

## 6. Co musi zamrozić ADR przed masową generacją

1. Listę klas, treści promptów (wersjonowane) i progi S2 (LCS, Jaccard, długości).
2. Zasadę mieszania populacji negatywów (syntetyczne per klasa vs `off_topic`
   organiczne) i limity per klasa — **przed** obejrzeniem pass-rate'ów.
3. Kryterium rozstrzygające: probe embedder na naturalnych zamrożonych
   zapytaniach; metadane klasowe służą do slice'ów, nie do selekcji post-hoc.
4. Deklarację wprost: strona syntetyczna to **wyłącznie negatywy** — żaden tekst
   teachera nie staje się `chosen` i nie trafia do continued SFT.
5. Wpis do rejestru wad znanych: pary syntetyczne uczą odróżniania od wad
   *wstrzykniętych*, które nie muszą mieć rozkładu wad *naturalnych* generatora.

## 7. Czego ten dokument nie robi

Nie buduje żadnej kohorty, nie zmienia par v3 ani trwających runów ramion, nie
otwiera zamkniętych kohort v4–v11, nie dotyka zbiorów testowych.
