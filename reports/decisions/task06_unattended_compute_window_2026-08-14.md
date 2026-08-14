# Okno obliczeń bezobsługowych 2026-08-14 (2–3 doby)

## Kontekst i decyzja właściciela

Właściciel wychodzi na 2–3 doby i zostawia maszynę włączoną (RTX 3060 Ti 8 GB,
15 GB RAM, 83 GB wolnego dysku), z polecenim maksymalnie efektywnego
wykorzystania sprzętu przy dwóch twardych warunkach: nie przesadzać z batch size
(ryzyko wyłączenia komputera) i nie dopuścić, by błąd tuż po wyjściu zmarnował
całe okno.

## Co jest wykonywane

Kolejka `configs/unattended_queue_2026-08-14.tsv`, nadzorowana przez
`scripts/run_unattended_queue.sh`:

1. **M-03 dla confirmu TriviaQA — seedy 45 i 46, oba ramiona** (4 treningi probe,
   ~14 min każdy). Rozszerzenie M-03 (decyzja właściciela 2026-08-13) podnosi
   liczbę seedów confirmu z 3 do 5 dla tanich korpusów (TriviaQA: 139782
   dokumenty ≤ 200 tys.). Uruchamiane są wyłącznie **treningi**; agregacja
   pięciu seedów i jakakolwiek reinterpretacja wyniku confirmu czekają na
   właściciela. Ramiona są symetryczne (W06 i Hybrid), więc żaden kierunek nie
   jest uprzywilejowany, a wynik zostanie policzony po wszystkich pięciu seedach
   niezależnie od tego, co pokaże.
2. **Trzy kohorty same-prompt v3/v4/v5, po 3000 pasaży** (~5,5 h każda) — skalowanie
   procedury, która właśnie przeszła w v2. Kohorty są zamrożone quality-blind
   *przed* startem okna i rozłączne klastrowo: nowe pole `cohort.partition`
   przypisuje każdy klaster near-duplicate do dokładnie jednej z trzech
   partycji, a wykluczenia obejmują smoke, pilot i v2. Weryfikacja na
   rzeczywistych danych: 3000/3000 unikalnych klastrów w każdej kohorcie, zero
   kolizji między wszystkimi sześcioma kohortami Task 06, a sumy pul
   (96994 + 96900 + 97018) odtwarzają legalną pulę pomniejszoną o v2.

3. **Sweep budżetu probe** (12 treningów, ~2,8 h) — diagnostyka do M-01 i M-03:
   te same dwa ramiona TriviaQA przy `--train-prefix-limit` 1024 i 2048 (pliki
   wejściowe mają dokładnie 3072 wiersze, więc mniejsze budżety są gwarantowane)
   na seedach 42/43/44. Odpowiada na pytanie, czy kierunek Hybrid–W06 utrzymuje
   się przy mniejszym budżecie danych probe, co jest wprost przesłanką dla M-01
   (predyktywność probe) i M-03 (stabilność seedów). Wyniki lądują w osobnym
   katalogu `runs/task05_probe_budget_sensitivity_v1` i **nie dotykają**
   artefaktów zakończonego confirmu ani jego agregatu.
4. **Kohorty v6–v11, po 3000 pasaży** (~33 h) — wypełnienie okna tą samą,
   sprawdzoną ścieżką. Łącznie z v3–v5 daje to 27000 nowych pasaży (216000
   kandydatów), rozłącznych klastrowo od wszystkiego wcześniejszego;
   zweryfikowano to na rzeczywistych danych dla wszystkich dwunastu kohort
   Task 06.

Razem ~53 h GPU, czyli okno 2–3 dób jest wypełnione, a nie tylko kilkanaście
godzin. Nadmiar jest nieszkodliwy: zadania są niezależne i wznawialne, więc
wcześniejszy powrót właściciela oznacza po prostu niedokończony ogon kolejki.
Koszt dyskowy całości to ~23 GB przy 83 GB wolnego i bramce 20 GB.

### Samonaprawianie (watchdog)

Aby awaria samego nadzorcy (OOM, zabity proces) albo **wyłączenie się komputera**
nie zmarnowały reszty okna, w cronie właściciela są dwa wpisy: wznowienie co 20
minut oraz wznowienie 3 minuty po restarcie systemu. Oba są opakowane w `flock`,
więc gdy kolejka działa, wznowienie jest no-opem (zweryfikowane: rc=3), a
znaczniki `done/<job>` gwarantują, że restart nie powtarza ukończonej pracy.

## Czego świadomie nie uruchamiam

- **Budowy par `chosen/rejected`** — wymaga zamrożenia polityki wag, progów i
  kalibracji komponentów. Jest to sedno Task 06 i jest tanie obliczeniowo
  (CPU, minuty), więc marnotrawstwem byłoby oddawać na to okno GPU, a
  ryzykiem — podejmować tę decyzję bez właściciela.
- **Treningu DPO (Task 07)** — `task07_training_authorized=false`, wymaga par i
  bramki audytu dual-LLM.
- **Pełnego benchmarku sędziów z Task 02** — `benchmark_rerankers.py` nie ma
  wznawiania i wymaga wejścia z dokładnie 10 negatywami na rekord, czego dla
  pełnego frozen dev (21241 rekordów) nie zweryfikowałem. Zadanie bez
  wznawiania i bez sprawdzonego wejścia nie wchodzi do okna bezobsługowego.
- **Ablacji teachera na Qwen3.6-27B Q4** — maszyna ma 15 GB RAM przy ~16 GB wag
  w Q4, więc na tym sprzęcie to nie jest realne.
- **Czegokolwiek dotykającego testów finalnych.** Nadzorca odrzuca komendy
  zawierające wzorce testów finalnych oraz `poweroff`/`shutdown`/`reboot`.

## Zabezpieczenia nadzorcy

- jedno zadanie naraz, po `flock`, i oczekiwanie na bezczynne GPU (do 2 h) —
  nigdy nie wchodzi w kolizję z innym procesem GPU;
- każde zadanie w osobnej grupie procesów z twardym limitem czasu; po limicie
  `TERM`, a po 60 s `KILL` na całą grupę, żeby nie został osierocony python
  trzymający GPU;
- ponowienia (2–3 próby): ponieważ wszystkie zadania są wznawialne, ponowienie
  kontynuuje pracę, a nie startuje od zera;
- awaria zadania **nie zatrzymuje kolejki** — jest zapisywana i kolejka idzie
  dalej;
- bramka dysku: przy mniej niż 20 GB wolnego kolejka kończy się czysto (koszt
  całego okna to ~3 GB, więc to margines, nie limit);
- cooldown między zadaniami przy temperaturze GPU ≥ 86 °C (do 30 min);
- batch pozostaje 8 dla generacji i scoringu oraz 2 dla treningu probe, zgodnie
  z zamrożonymi kontraktami; nic nie jest podnoszone „bo jest czas”;
- pamięć ukończonych zadań (`done/<job>`), więc ponowne uruchomienie kolejki
  jest bezpieczne i pomija to, co zrobione;
- `heartbeat.txt`, `queue.log`, `queue.events.jsonl` i `queue.summary.json`
  pozwalają po powrocie stwierdzić, co się stało, bez zgadywania.

Maszyneria nadzorcy została przetestowana na sztucznej kolejce: zadanie udane,
zadanie padające dwa razy i wznowione za trzecim, zadanie zawieszone ubite po
limicie, komenda z `poweroff` odrzucona, wiersz uszkodzony pominięty, drugi
przebieg pominął ukończone zadania. Ścieżki v3/v4/v5 zweryfikowano na
prawdziwych artefaktach do momentu ładowania modelu.

## Status po oknie

Kolejka produkuje wyłącznie artefakty pomiarowe. Żadna decyzja, promocja ani
status w rejestrze nie zmienia się automatycznie: wyniki (odsetki bramki,
trajektorie probe) opisuję po powrocie właściciela, wraz z decyzją o budowie par.
`final_tests_used=[]`.
