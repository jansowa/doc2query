# Kandydaci na model bazowy probe embeddera

## Stan decyzji

Recepta `probe_v1.yaml` pozostaje zamrożona dla porównywalności istniejących
artefaktów. Nie wykonano jeszcze pełnych runów, dlatego przed właściwą kampanią
probe należy wykonać krótki, identyczny benchmark kosztu i jakości modeli
poniżej, przypiąć revision i zapisać ADR wybierający jedną receptę v2.

## Kandydaci

| Model | Rola | Zaleta | Główne ryzyko |
|---|---|---|---|
| `sdadas/mmlw-roberta-base` | preferowany punkt odniesienia | polski encoder około 100M, już destylowany do embeddingów; mocny prior retrieval | większy koszt i istniejący prior embeddingowy mogą zmniejszać czułość probe'a na jakość danych syntetycznych |
| `sdadas/polish-distilroberta` | mały polski kontrolny MLM | około 82M, polski tokenizer i brak gotowego prioru retrieval | może wymagać większego budżetu, by ranking generatorów był stabilny |
| `jhu-clsp/ettin-encoder-32m` | tani kontrolny MLM | 32M i szybkie iteracje | angielski tokenizer/pretraining mogą nieadekwatnie karać polskie query |
| `jhu-clsp/ettin-encoder-17m` | najtańszy smoke/probe czułości | 17M, umożliwia wiele seedów | najwyższe ryzyko underfittingu i niestabilnego rankingu wariantów |

## Proponowana procedura wyboru

1. Dla każdego modelu: identyczne 10 tys. par naturalnych, 200 kroków, trzy
   seedy, ten sam test i budżet tokenów.
2. Odrzucić modele, które nie poprawiają się względem stanu początkowego albo
   mają niestabilny ranking seedów.
3. Na pozostałych porównać natural query z copy control i syntetycznymi query
   W03/W05/W06 na tej samej liczbie przykładów.
4. Preferować najmniejszy model, którego bootstrap i ranking wariantów zgadza
   się z mocniejszym polskim modelem.

`mmlw-roberta-base` wymaga prefiksu `zapytanie: ` dla query zgodnie z jego
model card. Obecny kod mean-pool nie obsługuje jeszcze asymetrycznych
prefiksów, więc nie wolno podmienić samej nazwy modelu w `probe_v1.yaml`.
