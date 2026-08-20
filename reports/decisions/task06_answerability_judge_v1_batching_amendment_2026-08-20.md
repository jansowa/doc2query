# Amendment do ADR V2-01: paczkowanie zapytań jednego pasażu

Status: **prospektywny amendment**, zamrożony przed uruchomieniem wariantu paczkowego na
jakimkolwiek itemie. Zmienia §3 ADR
[`task06_answerability_judge_v1.md`](task06_answerability_judge_v1.md) (protokół
wywołania) i wprowadza **nową wersję promptu**. **Kryteria akceptacji K1–K3 z §5
pozostają bez zmian**, podobnie jak model, pakiet itemów, źródła etykiet i konsekwencje
z §6. `final_tests_used=[]`, `used_for_pair_building=false`.

Poprzedni amendment (dekodowanie enumem):
[`task06_answerability_judge_v1_decoding_amendment_2026-08-20.md`](task06_answerability_judge_v1_decoding_amendment_2026-08-20.md).

## 1. Powód: prefill liczony N razy dla tego samego pasażu

Przy jednym zapytaniu na request każdy item wysyła **cały** prompt: system (561 znaków)
+ pasaż (mediana 331 znaków) + zapytanie. Pasaż i system prompt są jednak wspólne dla
wszystkich zapytań tego samego pasażu, a tych jest średnio **6,54** w puli autoryzowanej
i **6,64** w puli v4–v11. Zmierzone własności prefiksów są w
`artifacts/task06/judge_cache_diagnostics/`.

Liczenie na to, że serwer zamortyzuje ten narzut prefix cachem, **zawiodło w praktyce**:
operator raportuje `prefix cache hit rate` ≈ 2%, mimo że 84,7% requestów ma w swoim pasie
bezpośredniego poprzednika z identycznym pasażem. Przyczyna po stronie serwera pozostaje
nierozstrzygnięta (diagnostyka w `BRIEF.md` tamże), ale wniosek projektowy jest
niezależny od niej: **nie należy opierać kosztu na cudzej retencji bloków, jeśli ten sam
efekt można uzyskać strukturą requestu.**

## 2. Zamrożona zmiana protokołu

Nowa wersja promptu: **`task06-answerability-pl-v2-batched`**, SHA-256
`8b46fbbdd4fcf57cf1c3c35809c752a4f59f2ef91339a18bab0a02067e857d1f`. Wariant
pojedynczy `task06-answerability-pl-v1` (SHA-256
`74d3ee07757decbdf5655e1878c070f66bf05c90a60bf4dc5f56b1c520cfee84`) **pozostaje
nietknięty** i nadal jest przyrządem odniesienia.

- **Treść merytoryczna kryterium jest przepisana bez zmian** z wersji pojedynczej — to
  warunek porównywalności w bramce z §3. Dodane jest wyłącznie: wymuszenie oceny
  **niezależnej** dla każdego zapytania (z jawnym zakazem porównywania zapytań między
  sobą i zakładania, że mają różne werdykty) oraz format odpowiedzi z identyfikatorami.
- **Grupowanie**: paczka to **zawsze jeden pasaż**; mieszanie pasaży w paczce jest
  zabronione. Przydział pasaży do pasów pozostaje bez zmian (pasaż nigdy nie jest
  dzielony między pasy), a paczkowanie odbywa się **w obrębie pasa**. Współbieżności nie
  podnosimy — liczba requestów i tak spada kilkukrotnie.
- **Rozmiar paczki** jest parametrem (`--batch-size`), nie stałą w kodzie. Domyślną
  wartością pozostaje **1**, czyli przyrząd z ADR; wariant paczkowy wymaga jawnego
  włączenia. Pasaż z jednym zapytaniem używa tego samego formatu (tablica
  jednoelementowa) — jeden format, jedna wersja promptu.
- **Treść usera**: `{"passage": …, "queries": [{"id": 1, "query": …}, …]}` z `sort_keys`,
  więc pasaż stoi przed zapytaniami i paczki tego samego pasażu nadal mają wspólny
  prefiks. `id` są **lokalne dla paczki** (1..N); mapowanie `id → item_id` żyje po
  stronie klienta i nigdy nie jest wysyłane.
- **Schemat odpowiedzi** jest generowany per request, z `minItems == maxItems == N`,
  `id` jako liczba całkowita i `verdict` domknięty enumem `[yes, no, uncertain]`.
- **Budżet wyjścia**: patrz korekta w §2a. Po każdej odpowiedzi sprawdzane jest
  `finish_reason`.
- Bez zmian: `temperature = 0`, `seed = 20260817`, wyłączony thinking, model, pakiet.

## 2a. Korekta budżetu wyjścia (2026-08-20, po pierwszym runie paczkowym)

Pierwotnie zapisałem tu `12 · N + 20` tokenów i klauzulę, że `finish_reason=length` jest
błędem paczki „nie do naprawy podniesieniem budżetu w biegu". **Liczba była błędna**:
policzyłem ją dla zwięzłego JSON-a, a serwer formatuje odpowiedź z wcięciami — dla N=3 to
193 znaki zamiast 119, przy budżecie 56 tokenów. Skutek zmierzony u operatora: paczki
masowo kończyły się `length` i **spadały do fallbacku**, czyli na prompt pojedynczy.

To nie jest drobiazg wydajnościowy, bo psuje samą bramkę A/B: journal „kandydata"
wypełniał się wierszami przyrządu odniesienia, więc porównanie v1 vs v2 mierzyłoby v1
przeciw v1.

Zamrożona korekta:

- budżet to `tokens_per_verdict · N + overhead`, domyślnie **40 · N + 48** (dla N=8 daje
  368 tokenów wobec realnej potrzeby ~170, czyli ~2× zapasu). Oba składniki są
  parametrami CLI, nie stałymi w kodzie;
- przy `finish_reason=length` **ponowienie idzie z podwojonym budżetem** (z twardym
  sufitem 4096), a dopiero jego niepowodzenie uruchamia fallback.

Uzasadnienie odstąpienia od klauzuli „nie do naprawy w biegu": przy dekodowaniu zachłannym
(`temperature = 0`) `max_tokens` **nie zmienia rozkładu werdyktów** — decyduje wyłącznie o
tym, czy odpowiedź się zmieści. Truncation jest więc awarią wejścia/wyjścia, nie werdyktem
modelu, i jej naprawa nie jest dostrajaniem przyrządu pod wynik. Odwrotnie: **za mały
budżet aktywnie skaża pomiar**, bo systematycznie przenosi część itemów na inny przyrząd
przez fallback. Intencja pierwotnej klauzuli — zakaz zaklejania naruszeń **kontraktu**
(zbiór id, duplikaty, enum) podnoszeniem limitów — pozostaje w mocy i dotyczy §3.

Journale sprzed tej korekty (z wysokim udziałem fallbacku) nie wchodzą do bramki A/B;
zostają jako ślad, który tę korektę uzasadnia.

## 3. Walidacja odpowiedzi i fallback (zamrożone)

Odpowiedź paczki jest przyjęta tylko wtedy, gdy: jest poprawnym JSON-em, **zbiór `id`
jest dokładnie równy wysłanemu**, żadne `id` się nie powtarza i każdy werdykt należy do
enumu. **Kolejność elementów jest ignorowana** — mapowanie idzie wyłącznie po `id`.

Naruszenie: **jedno** ponowienie całej paczki, a przy drugim niepowodzeniu paczka jest
**rozbijana na pojedyncze requesty w formacie v1**, z oznaczeniem `fallback=true` w
każdym wierszu. Zdarzenie `batch_failed` z przyczyną i treścią odpowiedzi trafia do
journala jako dowód.

**Konsekwencja, którą trzeba czytać wprost:** fallback produkuje wiersze **innym
przyrządem** (prompt v1). Journal z wysokim udziałem fallbacku jest mieszanką, więc
import raportuje rozbicie po `prompt_version`, po `batch_size` i udział fallbacku, a
kalibracja liczona na takiej mieszance musi to podawać obok wyniku. Nie jest to obejście
zakazu mieszania przyrządów, bo fallback jest **jawnie oznaczony i policzony** — ale jego
udział jest liczbą, którą raport ma pokazać, a nie ukryć.

## 4. Bramka A/B (zamrożona przed odczytem)

Wariant paczkowy **nie zastępuje** wariantu pojedynczego, dopóki nie przejdzie
porównania na **tym samym zbiorze itemów** (pakiet kalibracyjny, 1540 itemów), przy
identycznym modelu, seedzie, temperaturze i dekodowaniu:

- **(B1) zgodność werdyktów per item ≥ 0,98**;
- **(B2) brak systematycznego dryfu**: dla każdej pary klas werdyktów migracje w obie
  strony muszą być zrównoważone. Operacyjnie: dokładny dwustronny test znakowy przy
  `p = 0,5` na liczbach przejść `i→j` i `j→i`, wykonany dla trzech par
  (`yes/no`, `yes/uncertain`, `no/uncertain`); **żadna** para nie może być istotna na
  poziomie 0,05.

B2 jest tu istotne, bo sama zgodność potrafi przepuścić dryf: 15 przejść `yes→no` na
1000 itemów daje 98,5% zgodności i jednocześnie jednostronną migrację. Test wyłapuje
dokładnie ten przypadek (pokryte testem CPU).

Niedowiezienie któregokolwiek warunku: zmniejsz `--batch-size` (8 → 6 → 4) i powtórz
porównanie; jeśli nie przechodzi także przy 4, **paczkowanie zostaje odrzucone** i
kalibracja oraz certyfikacja puli idą pojedynczymi requestami. Raport bramki zapisywany
jest do pliku (markdown + JSON), nie tylko na stdout.

Porównanie A/B jest **pomiarem przyrządu, nie kryterium jakości sędziego** — nie zmienia
ani nie zastępuje K1–K3.

## 5. Czego ten amendment nie robi

Nie zmienia kryteriów K1–K3, promptu pojedynczego, modelu, pakietu itemów, źródeł
etykiet ani progów. Nie podnosi współbieżności. Nie miesza pasaży w paczce. Nie zmienia
polityki par v1/v1.1 ani jej artefaktów. Nie autoryzuje treningu i nie otwiera testów
finalnych.
