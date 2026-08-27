# Amendment: audyt Groq schodzi z roli bramki dla par v3 (2026-08-27)

## Status

**Amendment do ADR** [`task06_judge_selected_pair_policy_v3.md`](task06_judge_selected_pair_policy_v3.md)
§8, spisany **przed jakimkolwiek audytem par v3**. Autoryzacja: decyzja właściciela z
2026-08-27, podjęta po przedstawieniu poniższych liczb.

Nie unieważnia żadnego zamkniętego audytu. Kontrakt
`task06-groq-dual-llm-preference-audit-v1` (prompt, rubryka, modele, limity) pozostaje
**bez zmian** — zmienia się wyłącznie rola, jaką ten audyt pełni w decyzjach.

## 1. Co mierzyliśmy, decydując

**Gęstość sygnału w audytach Groq.** Rozstrzygnięcie konsensusu wymaga, by oba modele
wskazały stronę; reszta to abstencja i nie wnosi nic do żadnej predykcji:

| audyt | próbka | rozstrzygnięte | udział |
|---|---|---|---|
| v2 | 500 par | 153 | **30,6%** |
| v2.1 | 800 par | 378 | **47,2%** |

Płacimy trzema oknami dziennych budżetów — w praktyce trzema dobami zegarowymi — za
150–380 informatywnych porównań. Powód jest w rubryce: jeden akapit bez definicji i z
zakazem rozumowania, przy 47,5% remisów u `gpt-oss`.

**Wartość krańcowa kolejnych audytów.** v1 wykazał, że porządkowanie marginesem jest
niecelne (zgodność 0,708/0,718 wobec 0,879 między sędziami) — wynik, który zabił
politykę. v2 wykazał, że oś B nie jest osią defektu (0,250 wobec 0,974) — wynik, który
zabił oś. v2.1 potwierdził znane (zgodność 45,12%, mechanizm) i zablokował się na progu
P3, który nigdy nie był wyprowadzony z pomiaru.

**Dostępność mocniejszego dowodu.** Kalibracja selektora v3 na **etykietach znanych z
konstrukcji** dała czystość **0,9793** przy jednomyślności, w 31 minut i bez limitów
dobowych ([raport](../measurements/task06_v3_selector_calibration_2026-08-27.md)).
Etykieta z konstrukcji nie pochodzi z żadnego modelu, więc jest dowodem mocniejszym niż
„inny model się zgadza".

## 2. Decyzja

**Audyt Groq przestaje być bramką dla par v3.** Predykcji P1–P4' ani ich następników
nie zamrażamy dla tej polityki, a zgoda sędziów Groq nie jest warunkiem użycia par.

W zamian obowiązują trzy rzeczy, w tej kolejności ważności:

1. **Kryterium rozstrzygające to probe embedder** na naturalnych zamrożonych
   zapytaniach (AGENTS.md §9.2: metryki powierzchniowe są pomocnicze). Pary v3 wchodzą
   do treningu DPO, a o ich wartości orzeka porównanie ramion, nie zgoda sędziów.
2. **Walidacja selektora na etykietach z konstrukcji** — wykonana, 0,9793, z jawnie
   zapisanym wąskim zakresem (ugruntowanie, kopiowanie, ogólność).
3. **Spot-check właściciela na 50 parach** z rzeczywistego rozkładu, ślepy, jako
   kontrola sanity przed treningiem. Pięćdziesiąt par oglądanych przez człowieka niesie
   więcej niż 800 ocen Groq, z których ponad połowa jest abstencją. **Nie jest to panel
   §9.3** i nie wolno tak tego raportować — to kontrola operacyjna, nie evidence.

## 3. Gdzie Groq zostaje

Audyt dual-LLM **nie jest wycofany z programu**. Zostaje przewidziany dla decyzji, w
której zewnętrzny głos faktycznie jest potrzebny: **przy zamrażaniu finalistów**
(Task 09/10), gdzie porównujemy procedury, a nie kalibrujemy selektor. Wtedy jednak
wymaga naprawy rubryki — obecna produkuje ponad połowę abstencji i to jest zapisany
warunek wstępny, nie sugestia.

## 4. Czego ten amendment nie zmienia

- Zamknięte audyty v1, v2 i v2.1 oraz ich wyniki pozostają zapisane bez reinterpretacji;
  bramka V2.1-05 pozostaje `INCONCLUSIVE`.
- Kontrakt Groq, jego prompt, rubryka, modele i limity — bez zmian.
- Zapas **1 453 nieoglądanych par v2.1** pozostaje nieoglądany; nie jest zwolniony do
  żadnego innego użycia tym dokumentem.
- Zakres ważności selektora v3 pozostaje wąski: bez formy i bez focusa.
- `task07_training_authorized=false` — zdjęcie Groq z roli bramki **nie jest**
  autoryzacją treningu. Ta pozostaje osobną decyzją właściciela.

`final_tests_used=[]`.
