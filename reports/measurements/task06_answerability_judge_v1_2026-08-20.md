# Pomiar: kalibracja sędziego odpowiadalności — **bramka K1–K3 przeszła**

Kontrakt: `task06-answerability-judge-v1`. ADR zamrażający kryteria **przed** odczytem:
[`task06_answerability_judge_v1.md`](../decisions/task06_answerability_judge_v1.md)
(+ amendmenty: [dekodowanie](../decisions/task06_answerability_judge_v1_decoding_amendment_2026-08-20.md),
[paczkowanie](../decisions/task06_answerability_judge_v1_batching_amendment_2026-08-20.md)).
Sędzia: `qwen3.8-27b` (`Qwen/Qwen3.8-27B-FP8`) na endpoincie vLLM 0.27.1 operatora.
Artefakt: `reports/measurements/task06/answerability_judge_v1/summary.json`.
Journal: `artifacts/task06/answerability_verdicts/verdicts_calib_single.jsonl`
(1540 werdyktów, prompt `task06-answerability-pl-v1`, dekodowanie `json_schema_enum`).
`human_evidence_claimed=false`, `used_for_pair_building=false`,
`task07_training_authorized=false`, `final_tests_used=[]`.

## Wynik: sędzia przyjęty

| kryterium | wymóg z ADR | zmierzone | werdykt |
|---|---|---|---|
| **K1** accuracy na konsensusie Groq | ≥ 0,85 | **0,8566** (n=809) | ✅ |
| **K1** balanced accuracy | ≥ 0,75 | **0,8878** | ✅ |
| **K2** `ungrounded` → `no` | ≥ 0,80 | **1,0000** (180/180) | ✅ |
| **K2** `good_specific` → `no` | ≤ 0,20 | 0,1056 | ✅ |
| **K2** `good_alternative` → `no` | ≤ 0,20 | 0,1222 | ✅ |
| **K3** udział `uncertain` | ≤ 0,25 | **0,0065** | ✅ |

Status artefaktu: `accepted_as_axis_a_answerability_signal`. Rozkład werdyktów:
`yes` 910, `no` 620, `uncertain` 10. Zero zdarzeń `out_of_schema`, wszystkie werdykty
przy **pierwszej** próbie — poprawka dekodowania enumem wyeliminowała defekt
`{"verdict": "verdict"}` całkowicie (przed nią ~0,3% itemów traciło werdykt).

Raportowane, **nie bramkowane**: zgodność z pojedynczymi sędziami Groq — `gpt-oss-120b`
0,7812 (CI [0,7530; 0,8075]), `qwen3.6-27b` 0,8004 (CI [0,7742; 0,8236]); z konsensusem
0,8566 (CI [0,8331; 0,8801]). `recall_yes` 0,8328, `recall_no` **0,9429**.

## Powtarzalność przyrządu: 0,9909, i to jest samodzielny wynik

Operator dostarczył **dwa** journale wyprodukowane tym samym przyrządem (ten sam model,
prompt, `seed`, `temperature=0`, dekodowanie) na tych samych 1540 itemach. Zgodność
między nimi: **0,9909** — 14 różnic, w obu kierunkach (`no→yes` 6, `uncertain→no` 4,
`yes→no` 3, `no→uncertain` 1).

To potwierdza wcześniejszą obserwację, że **serwer nie jest deterministyczny** mimo
`temperature=0` i przypiętego seeda: continuous batching zmienia skład batcha, a więc
numerykę. Dwie konsekwencje:

1. **Bramka jest stabilna.** Drugi run daje K1 accuracy 0,8557 (wobec 0,8566),
   balanced 0,8852 (0,8878), `ungrounded` → `no` znów 1,0000, `uncertain` 0,45% —
   **oba runy przechodzą**. Rozrzut K1 między runami to 0,09 pp, czyli 7× mniej niż
   zapas nad progiem (0,66 pp).
2. **Sufit szumu jest skalą dla bramki A/B paczkowania.** Zapas między progiem 0,98 a
   powtarzalnością 0,991 to ~1,1 pp, więc wariant różniący się od pojedynczego w granicach
   szumu byłby nierozstrzygalny. Zmierzona zgodność paczkowego wyniosła jednak **0,9052**,
   czyli ~10× poza szumem — wynik tamtej bramki jest więc jednoznaczny, a nie graniczny.

## Uczciwe granice wyniku K1

Zapas nad progiem accuracy jest **mały** (0,8566 wobec 0,85). Kryterium nie zostało
dobrane pod ten wynik — próg zamrożono 2026-08-19, przed wyprodukowaniem jakiegokolwiek
werdyktu — ale trzeba powiedzieć wprost, że gdyby próg wynosił 0,86, sędzia by nie
przeszedł. Mocniejszą przesłanką jest **balanced accuracy 0,8878** przy `recall_no`
0,9429: sędzia faktycznie wykrywa klasę „nieodpowiadalne", a nie tylko dziedziczy bazę
klasy większościowej (0,7846). Dla porównania oblane proxy leksykalne miało na swoim
holdoucie balanced accuracy 0,710.

Nie wolno też czytać 0,8566 jako „85,7% poprawności": referencją jest **konsensus dwóch
innych modeli**, nie prawda absolutna. Sami sędziowie Groq zgadzali się ze sobą co do
odpowiadalności w 0,817 na **wszystkich** stronach (817/1000), a K1 jest liczone na
podzbiorze konsensusowym, czyli łatwiejszym. Te dwie liczby nie są wprost porównywalne.

## Przegląd ręczny (zobowiązanie z §5 ADR)

Przejrzałem **34 werdykty** stratyfikowane deterministycznie po haszu `item_id`: 10 z
grupy „sędzia `yes` / konsensus `no`", 8 z „sędzia `no` / konsensus `yes`", 6 z klas
`good_*` ocenionych jako `no` (nadmierne odrzucanie) oraz wszystkie 10 `uncertain`.
Zastrzeżenie z ADR obowiązuje: przegląd mógł kalibrację wyłącznie **unieważnić**.
Nie unieważnił, a obraz jest odwrotny do oczekiwanego:

- **`yes` sędziego wobec `no` konsensusu (10 przypadków)**: w 8 werdykt sędziego jest
  bardziej obronny. Pasaż o CDP Mayer podaje hrabstwo (`Yavapai`), pasaż o zielu wierzby
  podaje nazwę (`Epilobium parviflorum`), pasaż o beri-beri wymienia dotknięte układy.
  Zapytania są w tych przypadkach pokaleczonymi tłumaczeniami („definicja redding, cal w
  ka", „jakie hrabstwo to De Chaineted Mayer AZ") i sędziowie Groq wydają się karać
  formę zapytania, podczas gdy zamrożony prompt pyta wyłącznie o to, czy odpowiedzi da
  się udzielić z pasażu.
- **`no` sędziego wobec `yes` konsensusu (8)**: w 5 sędzia ma rację lub jest obronny —
  pasaż o pensji gubernatora Missisipi podaje **średnią i medianę krajową, ale nie kwotę
  stanową**; „jak używać numeru VIN" przy pasażu wyjaśniającym, czym VIN jest;
  „ile mil robi autobus w ciągu roku" przy pasażu o 12-letniej żywotności. Realne
  pomyłki sędziego to ok. 2 (mężowie Scarlett O'Hara, „kim jest velcro").
- **Klasy `good_*` ocenione jako `no` (6)**: w 4 przypadkach **etykieta z konstrukcji
  jest zbyt optymistyczna, a sędzia ma rację** — pasaż o Elementary OS nie podaje
  licencji, pasaż o Christinie Ricci nie podaje zarobków z okresu dziecięcego, pasaż o
  joint venture nie wymienia firm, a jeden item ma pasaż o zespole gruszkowatym przy
  zapytaniu o medycynę osteopatyczną (artefakt konstrukcji korpusu). Oznacza to, że
  zmierzone 10,6% i 12,2% „nadmiernych odrzuceń" jest w większości **szumem etykiet**,
  nie błędem sędziego — czyli zapas w K2 jest bezpieczniejszy, niż wygląda.
- **Wszystkie 10 `uncertain`**: autentyczne przypadki częściowej odpowiedzi (czas
  zadomowienia leylandii „zależy od warunków", koszt naprawy manetek „zależy od
  rodzaju"). Abstencja jest trafnie umieszczona i wynosi 0,65%, więc nie zabiera podaży.

**Czego przegląd nie uprawnia.** Powyższe nie podnosi zmierzonych liczb i nie może być
użyte do rozluźnienia progów. Jest natomiast przesłanką, że **referencja jest słabszą
stroną tego porównania** — i to jest argument metodologiczny na przyszłość (jeśli kiedyś
potrzebna będzie mocniejsza referencja, trzeba panelu ludzkiego, nie kolejnego LLM-a), a
nie poprawka do dzisiejszego wyniku.

## Wykryty kierunek obciążenia (do uwzględnienia w ADR V2-03)

`recall_no` 0,9429 przy `recall_yes` 0,8328 znaczy, że sędzia jest **surowszy** niż
konsensus: częściej mówi `no`. W roli filtra strony `chosen` osi A oznacza to filtr
**konserwatywny** — będzie odrzucał część kandydatów faktycznie odpowiadalnych, kosztem
podaży, ale nie kosztem czystości. Przy 21102 grupach z czystym `chosen` (inwentarz
V2-00) jest na to zapas, ale ADR V2-03 musi policzyć realną podaż **po** certyfikacji
puli, a nie założyć ją z góry.

## Co to odblokowuje, a czego nie

- Oś A polityki par v2 **ma** sygnał odpowiadalności. Luka zmierzona w audycie v1
  (16,6%/18,8% nieodpowiadalnych `chosen`) przestaje być nieadresowana, a dług nazwany po
  porażce proxy leksykalnego jest spłacony.
- **Nie** jest jeszcze wykonana certyfikacja puli kandydatów (23 676 itemów w kohortach
  autoryzowanych, 148 619 w v4–v11) — bez niej nie da się zbudować par osi A.
- **Bramka A/B paczkowania: ODRZUCONA** (szczegóły niżej). Certyfikacja puli idzie
  przyrządem pojedynczym, który przeszedł K1–K3.
- Progów nie zmieniono, rubryk nie zmieniono, żadnej pary nie zbudowano,
  `final_tests_used=[]`.

## Bramka A/B paczkowania: odrzucona, i to podwójnie

Raport: `reports/measurements/task06/judge_batching_ab_v1/`. Porównanie na tych samych
1540 itemach, ten sam model, `seed`, `temperature`, dekodowanie; journal kandydata jest
czysty (0 fallbacku, 100% promptu `task06-answerability-pl-v2-batched`), więc wynik nie
jest artefaktem wcześniejszego błędu budżetu tokenów.

**Nie ma zysku wydajnościowego.** Zmierzone przepustowości na 1540 itemach: pojedynczo
**19,1 it/s** (1:20) i 17,0 it/s (1:30) w dwóch runach, paczkami **16,3 it/s** (1:34).
Paczkowanie było *marginalnie wolniejsze*. Pule liczone pojedynczo trzymają 19,2–19,5
it/s. Wniosek: ten serwer jest ograniczony **dekodowaniem, nie prefillem** — 665
requestów paczkowych zajęło 94 s, czyli 2,25 s na request przy średnio 2,32 itemu
(0,97 s/item wobec 0,84 s/item pojedynczo). Mój wcześniejszy szacunek 3–5× przyspieszenia
był **błędny**: opierał się na założeniu, że wąskim gardłem jest powtarzany prefill
pasażu. Obala to też sens wariantu hybrydowego (wiele pasaży w requeście) — premisa jest
ta sama.

**Kryteria bramki nie przeszły.**

| kryterium | wymóg | zmierzone | werdykt |
|---|---|---|---|
| B1 zgodność per item | ≥ 0,98 | **0,9052** | ❌ |
| B2 brak dryfu | żadna para nieistotna | **dryf istotny** | ❌ |

Dryf jest jednostronny i celuje w abstencję: `yes→uncertain` 21 wobec 2 w drugą stronę
(p = 0,0001), `no→uncertain` 19 wobec 2 (p = 0,0002); udział `uncertain` rośnie z 0,65%
do 2,99%. Para `yes/no` jest zrównoważona (52 vs 50, p = 0,92), ale obejmuje 102 itemy,
czyli 6,6% — przy własnej powtarzalności przyrządu 0,9909 (ok. 0,9% przeskoków) to
**rząd wielkości więcej niż szum**.

Interpretacja mechanizmu: oceniając kilka zapytań w jednym kontekście, model **częściej
się waha**. Kontrakt traktuje `uncertain` jako blokadę roli `chosen`, więc paczkowanie
nie tyle myliłoby werdykty, ile systematycznie **zjadałoby podaż** osi A.

Decyzja: **paczkowanie odrzucone**, `--batch-size` pozostaje na domyślnym 1. Prompt
`task06-answerability-pl-v2-batched` zostaje w kodzie jako odrzucony wariant z zapisanym
wynikiem, nie jako opcja do włączenia „gdy się przyda". Gdyby kiedyś wrócił, wymaga nowej
bramki — próg 0,98 przy suficie szumu 0,991 zostawia ~1,1 pp zapasu, więc każdy przyszły
wariant musi być praktycznie nieodróżnialny od pojedynczego.
