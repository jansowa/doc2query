# Amendment do ADR V2-01: domknięcie wartości werdyktu schematem (dekodowanie)

Status: **prospektywny amendment**, zamrożony przed policzeniem jakiejkolwiek
zgodności werdyktów z etykietami. Zmienia **wyłącznie** §3 ADR
[`task06_answerability_judge_v1.md`](task06_answerability_judge_v1.md)
(protokół wywołania). **Kryteria akceptacji K1–K3 z §5 pozostają bez zmian**,
podobnie jak prompt, zbiór itemów, źródła etykiet i konsekwencje z §6.
`final_tests_used=[]`, `used_for_pair_building=false`.

## 1. Co zmierzono

Pierwszy run kalibracyjny (2026-08-20, endpoint vLLM operatora,
`qwen3.8-27b` = `Qwen/Qwen3.8-27B-FP8`) ujawnił, że sędzia zwraca czasem
**`{"verdict": "verdict"}`** — składniowo poprawny JSON, w którym w miejscu
wartości powtórzona jest nazwa pola. Odsetek: rzędu **0,3% ocenionych itemów**
(pojedyncze itemy na ~1100 w logu operatora). Przy `--invalid-retries 3` część
takich itemów wraca poprawnie w kolejnej próbie — serwer **nie jest
deterministyczny**, bo continuous batching zmienia skład batcha, a więc i
numerykę. Reszta kończy jako item **bez werdyktu**.

Sprawdzone i **odrzucone** hipotezy o przyczynie (na 11 itemach z logu):

- **długość pasażu**: mediana 332 znaki w próbce wobec 331 w całym pakiecie —
  brak efektu;
- **znaki kontrolne w pasażu** (U+0080…U+0099, U+2028 i podobne): 2 z 11
  itemów, przy 141 z 1540 itemów pakietu je zawierających — najwyżej słaby
  związek, nierozstrzygalny na tej liczbie;
- **koncentracja na trudnych przypadkach**: udział etykiet „trudnych”
  (konsensus `no` + strony sporne) to 27% w próbce wobec 23% w pakiecie —
  **żadnego sygnału**. Wcześniejsza intuicja, że to przypadki graniczne, nie
  wytrzymała konfrontacji z liczbami i jest tu zapisana jako odrzucona.

Systematycznego wyzwalacza więc **nie zidentyfikowano** i ten amendment go nie
zakłada.

## 2. Dlaczego to jednak jest defekt przyrządu, niezależnie od przyczyny

`response_format: {"type": "json_object"}` wymusza **poprawną składnię JSON, ale
nie zbiór wartości**. Skutek jest asymetryczny i szkodliwy:

- zamrożona przestrzeń decyzyjna to `[yes, no, uncertain]`, gdzie `uncertain`
  istnieje **właśnie** na wypadek niepewności sędziego (§3 ADR: `uncertain`
  blokuje rolę `chosen`, ale nie jest defektem);
- obecne dekodowanie pozwala modelowi **wyjść z tej przestrzeni w całości**, a
  wtedy item nie dostaje `uncertain`, tylko **przepada** — zamiast obserwacji
  „nie wiem” dostajemy dziurę w danych;
- dziury nie są neutralne dla mianowników K1–K3. Nie wykazano, że są losowe
  (patrz §1: brak identyfikacji wyzwalacza to **nie** dowód losowości), więc
  liczenie kryteriów na przetrzebionym zbiorze niesie nieznane obciążenie.

Domknięcie wartości schematem **nie zmienia kontraktu decyzyjnego** — wymusza
dokładnie to, co kontrakt i tak zakłada — a zamienia dziury w obserwacje
należące do zamrożonej przestrzeni.

## 3. Zamrożona zmiana

`response_format` przyjmuje wariant `json_schema_enum`, wysyłany jako:

```json
{"type": "json_schema",
 "json_schema": {"name": "answerability_verdict", "strict": true,
   "schema": {"type": "object",
     "properties": {"verdict": {"type": "string",
                                "enum": ["yes", "no", "uncertain"]}},
     "required": ["verdict"], "additionalProperties": false}}}
```

Bez zmian pozostają: prompt (`task06-answerability-pl-v1`, SHA-256
`74d3ee07757decbdf5655e1878c070f66bf05c90a60bf4dc5f56b1c520cfee84`),
`temperature = 0`, `seed = 20260817`, `max_tokens = 24`, wyłączony thinking,
model, pakiet itemów i **wszystkie progi**.

**Zakaz mieszania.** Wariant dekodowania jest zapisywany w **każdym** wierszu
journala (pole `decoding`), a runner odmawia dopisywania do journala
wyprodukowanego innym wariantem — łącznie z journalami sprzed tego amendmentu,
które pola nie mają i są traktowane jako `json_object_przed_amendmentem`.
Kalibracja i certyfikacja puli muszą używać **tego samego** wariantu: sędzia
skalibrowany przyrządem A nie może filtrować przyrządem B.

## 4. Co się dzieje z pracą wykonaną przed amendmentem

Runy z 2026-08-20 pod `json_object` (kalibracja oraz komplet 23 676 itemów puli
autoryzowanej i ~8% puli v4–v11) **nie są usuwane**. Są zmierzonym dowodem, na
którym opiera się §1 tego amendmentu, i zostają jako ślad audytowy pod
własnymi nazwami. **Nie wchodzą** natomiast do kalibracji ani do filtrowania osi
A — całość powstaje od nowa pod `json_schema_enum`. Koszt przeliczenia jest
znany i mały: zmierzone 16,5 itemu/s daje ~2 min na pakiet kalibracyjny, ~25 min
na pulę autoryzowaną i ~2,5 h na v4–v11.

Porównanie obu wariantów na tym samym zbiorze itemów jest **dopuszczalnym
produktem ubocznym** (ile itemów zmieniło werdykt, ile dziur zniknęło) i będzie
raportowane, ale **nie jest** kryterium akceptacji i nie może wpłynąć na progi.

## 5. Czego ten amendment nie robi

Nie zmienia promptu — jego znanym ograniczeniem pozostaje to, że przykład
`{"verdict": "yes"}` podaje jako pierwszą wartość `yes`, co może działać jak
zakotwiczenie; zmiana promptu byłaby nową wersją przyrządu i wymagałaby
osobnego, prerejestrowanego runu, nie poprawki w biegu. Nie zmienia K1–K3, nie
zmienia źródeł etykiet, nie autoryzuje treningu, nie otwiera testów finalnych i
nie dotyka polityki par v1/v1.1 ani jej artefaktów.
