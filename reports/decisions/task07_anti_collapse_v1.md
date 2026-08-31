# ADR: przeciwdziałanie kolapsowi generacji po DPO, v1 (2026-08-31)

## Status

**Prospektywny ADR, zamrożony przed uruchomieniem któregokolwiek z ramion.**
Kontekst: [pomiar kolapsu](../measurements/task07_generation_collapse_2026-08-31.md)
— ramiona DPO tracą 22–41% slotów generacji na duplikaty przy 4,5% punktu
startowego, a kolejność strat pokrywa się z kolejnością kosztu NLL na dev.
Autoryzacja: polecenie właściciela z 2026-08-31 („zastanów się, jak ograniczyć
lub rozwiązać ten problem"). `final_tests_used=[]`.

## 1. Diagnoza mechanizmu (trzy współdziałające przyczyny)

1. **Strata**: sigmoid DPO nagradza różnicę logprobów, więc najtańszą drogą jest
   spychanie `rejected` — a to zabiera masę prawdopodobieństwa całemu rozkładowi
   wyjścia. Entropia spada, sampling (T=0,8, top-p 0,95) trafia w ten sam tryb.
2. **Dane**: w kohorcie defect grupa wystawia do trzech par o **identycznym
   `chosen`** (różne klasy wad dzielą zwycięzcę turnieju). Model widzi „dla tego
   pasażu jest dokładnie jedna dobra odpowiedź" po kilka razy — trening wprost
   zachęca do jednej odpowiedzi na pasaż.
3. **Kontrolki**: żadna para nie karze złamania kontraktu Forma/Intencja, więc
   zależność wyjścia od kontrolek nie jest wzmacniana; po zawężeniu rozkładu
   wyjście przestaje na nie reagować.

## 2. Zamrożone interwencje (każda celuje w inną przyczynę)

| ramię | interwencja | cel | parametry zamrożone |
|---|---|---|---|
| `EQ102` | bottom DPO ucięte do 102 kroków | kontrola konfundenta kroków, nie lek | plan bottom bez zmian, `max_steps=102` |
| `BETA02` | defect DPO z beta 0,2 | mocniejsza kotwica KL przy referencji | nowy plan `task07-dpo-plan-defect-beta02-s42`; beta 0,2 jest w prerejestrowanej liście ablacji zadania |
| `RPO` | defect DPO + regularyzator NLL na `chosen` | bezpośrednio trzyma prawdopodobieństwo `chosen` | `loss = DPO + λ·NLL_token(chosen)`, **λ = 1,0**; to jest JEDYNY dopuszczony alternatywny loss (zadanie pozwala na „co najwyżej jeden alternatywny stabilny loss po baseline"; baseline sigmoid jest policzony) |
| `DIVCH` | defect DPO na kohorcie z **różnicowanym `chosen`** | usuwa sygnał „jedna odpowiedź na pasaż" | w grupie o k parach klasy dostają **różne** `chosen` z kandydatów `ok`+answerable TAK (rotacja deterministyczna po posortowanych id); pary bez drugiego kandydata zachowują zwycięzcę |
| `wrong_form` (serwer) | nowa klasa par: treść dobra, złamany kontrakt formy | uczy zależności od kontrolek | mutacja formy `chosen` (pytanie↔fraza); etykieta z konstrukcji, weryfikacja regexem, bez LLM |

λ=1,0 dobrane przed pomiarem z rzędów wielkości: startowy DPO loss ≈0,69 i
NLL/token ≈0,62 są porównywalne, więc λ=1,0 daje obu członom zbliżoną wagę.
Nie będzie strojone po obejrzeniu wyników; zmiana wymaga amendmentu.

## 3. Miary sukcesu (zamrożone przed runami)

Dla każdego ramienia, w tej kolejności ważności:

1. **Różnorodność generacji** na kohorcie ekranowej probe (496 pasaży × 4
   kontrolki, identyczny protokół jak pomiar kolapsu): udział duplikatów i
   liczba wypełnionych slotów. Sukces interwencji = duplikaty **≤ 12%**
   (połowa dystansu między defect DPO 22,2% a kontrolami ~8%); zapis wyniku
   obowiązuje niezależnie od tego, czy próg padł.
2. **Trafność marginesu na dev** — nie może spaść poniżej wartości continued
   SFT-równoważnej; ramię, które leczy kolaps gubiąc cały sygnał preferencji,
   nie jest lekiem.
3. NLL/token na `chosen` — raportowane, bez progu.

Rozstrzyga, jak zawsze, probe embedder; te miary wybierają, **które ramiona
w ogóle warto** do probe wystawić.

## 4. Koszty i kolejność wykonania

GPU (sekwencyjnie, po zakończeniu trwającej generacji probe): EQ102 (~40 min) →
BETA02 (precompute ~10 min + trening ~65 min) → RPO (~65 min) → DIVCH (handoff +
precompute ~15 min + trening ~65 min) → generacja różnorodności dla BETA02, RPO,
DIVCH (3 × ~60–90 min). Dla EQ102 generacji nie robimy — to kontrola NLL, nie
kandydat.

Rezygnacja: generacje probe ramion **weighted SFT** wypadają z kolejki (dwa
punkty jeszcze niepoliczone). Uzasadnienie zmierzone: wsft ≈ csft na dev
(ΔNLL ≤ 0,001 w obu kohortach), a zwolnione ~1,8 h idzie na ramiona
antykolapsowe. Wygenerować je można później jedną komendą.

Serwer (paczka v2): `wrong_form` dla 2 362 grup + zaległa `lexical_mutation`
dla 953 grup ≈ 5,3 tys. wywołań.

## 5. Czego ten ADR nie zmienia

Autoryzacja treningu obejmuje te ramiona jako ablacje na już autoryzowanych
kohortach (bottom, defect); kohorta DIVCH to ta sama pula par defect z inną
regułą wyboru `chosen` spośród już zweryfikowanych kandydatów — nie dodaje
żadnego nowego tekstu. Testy zamknięte, pule v4–v11 zamknięte, `answer_leak`
odłożone do wyniku v2.
