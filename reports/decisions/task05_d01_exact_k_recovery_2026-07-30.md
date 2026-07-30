# ADR: techniczna wspólna kohorta exact-K po D01

## Status i problem

Decyzja przyjęta 30 lipca 2026 przed finalną materializacją kohorty czterech
ramion. Obie generacje D01 ukończyły pełny frozen `dev_intrinsic_rank10`, ale
bounded retry pozostawił 6 exhausted groups dla 1.5B i 8 dla 4.5B. Grupy są
rozłączne. Oryginalne artefakty zawierają odpowiednio 26386 i 26384 query,
więc nie spełniają exact K=4 i nie mogą wejść bezpośrednio do comparatora ani
materializatora probe.

## Rozważone rozwiązania

1. Deterministyczny top-up tylko brakujących kontrolek. Zachowałby wszystkie
   passage, ale podnosiłby retry ceiling wyłącznie dla historycznych ramion i
   wymagał ponownego ładowania obu generatorów. Analogiczny ceiling należałoby
   następnie zastosować do baseline'ów, co zwiększa i komplikuje budżet.
2. Wspólna kohorta exact-K wybrana wyłącznie przez techniczną kompletność
   outputów. Nie generuje dodatkowych prób, zachowuje wspólny ceiling i pozwala
   zastosować identyczną kolejność do wszystkich ramion.

Wybrano wariant 2. Po ukończeniu pełnych matched generation W05 i rzeczywistego
W06 BS8 pipeline wyznaczy przecięcie grup mających dokładnie cztery unikalne,
poprawne query we wszystkich czterech ramionach. Wstępny górny limit, wynikający
wyłącznie z dwóch gotowych D01, to 6584 passage i 26336 par; finalna liczba może
być mniejsza, jeśli bounded retry wyczerpie się także w baseline. Nie wolno
wpisać jej przed audytem baseline'ów.

## Kontrakt recovery

- wybór nie odczytuje score'ów primary, shadow, corpus ani innych metryk
  jakości;
- zachowana zostaje frozen kolejność `evaluation_group_id` i oryginalny indeks
  grupy używany w seedzie;
- `K=4`, base seed `42`, group stride `1000`, attempt stride `1`;
- `max_new_tokens=64` i maksymalnie 3 próby na kontrolkę/query;
- token ceiling wynosi `passage_count × 4 × 64 × 3` i będzie zapisany po
  finalnym przecięciu;
- usunięta grupa otrzymuje jawny powód i listę niekompletnych ramion;
- oryginalne JSONL, journale, identity i summary pozostają bez zmian;
- odzyskane kopie mają osobne identity, fingerprint selekcji i provenance
  źródłowych SHA-256;
- `final_tests_used=[]`.

Polityka minimalizuje bias przez użycie jednego, mechanicznego kryterium
kompletności dla całej czwórki. Nie jest wynikiem jakościowym i nie może służyć
do promocji generatora. Po intrinsic guardrails nadal wymagany jest
porównywalny probe embeddera; jego trening nie należy do tej sesji.
