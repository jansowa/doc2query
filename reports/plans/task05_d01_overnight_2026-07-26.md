# Plan nocnego runu Task 05 D01

## Cel

Pierwsze treningi z jawną kontrolą formy i intencji, bez focusu:
`D01-1.5B-STYLE-50K-S42` i `D01-4.5B-STYLE-50K-S42`. Runy używają dokładnie
tego samego naturalnego subsetu 50k co W05/W06 i odpowiadających im
sprawdzonych recept QLoRA, zmieniając prompt treningowy na kontrolowany.

Nie jest to pełne rozstrzygnięcie D01: po treningu nadal są wymagane intrinsic
guardraile, porównywalny probe embeddera, bootstrap oraz audyt człowieka.

## Uzasadnienie budżetu

W05 1.5B na 50k przykładów, max length 512 i LoRA r=8 trwał
`31 868.68 s` (8 h 51 min), osiągnął `1.569 examples/s` i peak reserved
`1 994 391 552 B`. W06 4.5B na tym samym subsetcie trwał `29 653.01 s`
(8 h 14 min), osiągnął `1.686 examples/s` i peak reserved `7 736 393 728 B`.
Łączny empiryczny czas obu treningów to około 17 h 05 min. Smoke i dwa
diagnostyczne panele po 500 dev passages × 4 kontrolki powinny pozostawić
kilkugodzinny bufor w oknie 24 h; czasu kontrolowanej generacji nie zmierzono
jeszcze dla tych checkpointów.

Historyczne panele po jednym greedy query wskazują około 25 minut dla 2 000
generacji 1.5B i 62 min dla 2 000 generacji 4.5B. Kontrolowane sampling/retry
może ten czas zwiększyć, dlatego operacyjny szacunek całej kolejki wynosi
19–21 godzin, z około 3–5 godzinami bufora.

Nie jest to czysta ablacja skali: 1.5B jest checkpointem base, a 4.5B
checkpointem instruct. Każdy D01 jest przede wszystkim dopasowaną ablacją
kontrolek względem własnego W05/W06; porównanie 1.5B↔4.5B pozostaje
diagnostyczne do czasu osobnej kontroli base/instruct i probe.

## Kontrakt danych

CPU preflight rzeczywiście przygotował:

- train: 50 000;
- dev: 1 000;
- dataset fingerprint: `017a26ebcf6c5811d5c84498d44881d943c919680e9eed482a649409dfc06b73`,
  identyczny jak ukończony W05;
- formy train: 31 628 `full_question`, 14 545 `keyword_query`, 3 827 `unknown`;
- intencje train: 26 371 `fact_lookup`, 12 468 `definition`, 3 879
  `entity_lookup`, 3 197 `procedure`, 351 `comparison`, 3 734 `unknown`.

`unknown` jest jawnym abstention reguł, a nie etykietą wymuszoną na jedną z
klas. Kalibracja per domena i ręczny audyt etykiet pozostają niewykonane.

## Runner

```bash
bash scripts/run_task05_d01_overnight.sh
```

Runner:

1. wymaga ukończonego, inference-only/dev-only artefaktu pełnej bramki HN;
2. sprawdza CUDA 12.4, sześć konfiguracji i co najmniej 20 GiB wolnego miejsca;
3. wykonuje osobny, wznawialny 3-step smoke dla 1.5B i 4.5B;
4. wykonuje oba wznawialne treningi 50k przed panelami diagnostycznymi;
5. generuje osobne diagnostyczne panele z naturalnego dev;
6. nie odczytuje żadnego finalnego testu.

Log: `logs/task05_d01_overnight/queue.log`.

Status kroków: `reports/measurements/task05_d01_overnight/status.tsv`.

## Bramka poprzedzająca

Pełna HN gate zakończyła się jako `measured` na 775 wspólnych legalnych query,
artifact fingerprint
`bc02f475c5955f32c92612519d35a21804db9ad2884426b788c17afee0d660a9`.
Decyzja utrzymuje HN0+filter/drop i nie otwiera testów finalnych.
