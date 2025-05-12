## Struktura repozitorija
- ```models/``` mapa z pretrained modeli za ensemble
- ```data/``` mapa za podatke
- ```embeddings/``` mapa za SloBerta embeddine podatkov
- ```predstavitev/``` mapa z datotekami za predstavitev
- ```workspace/``` mapa z datotekami razvojnega okolja (ne predstavlja dela končne oddaje)
- ```final.py``` datoteka za zagon modela za napovedovanje števila komentarjev
- ```predstavitev.pdf``` pdf datoteka s predstavitvijo
- ```requirements.txt``` datoteka s potrebnimi knjižnicami 

## Zagon vizualizacije in vzpostavitev okolja
Vizualizacija je namenjena zagonu v `python 3.12`. Knjižnice, ki so potrebne za zagon namestimo z
```bash
pip install -r requirements.txt
```
Skripta `final.py` brez dodatnih argumentov zažene napoved na testni množici `data/rtvslo_test.json` in na SloBerta vložitvah `embeddings/sloberta_embeddings_test.pt`. Z uporabo dodatnih argumentov lahko določimo poljubno testno množico in vložitve, naredimo Sloberta embedding ali streniramo nov model za ensemble.

Argumenti:
- `--train`: streniraj model `model_01.pt` in ga shrani med modele v `models/`
- `--embed`: način za embedding
- `--data_path`: pot do json datoteke s podatki
- `--emb_path`: pot do SloBerta embeddingov podatkov

Primeri:
```bash
python final.py

python final.py --train

python final.py --embed --data_path test_data.json --emb_path embeddings/test_data.pt

python final.py --data_path test_data.json --emb_path embeddings/test_data.pt
```