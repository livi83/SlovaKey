# SlovaKey

SlovaKey je knižnica pre extrakciu kľúčových slov z textov v slovenčine. 

## Inštalácia

pip install -r requirements.txt

## Použitie 
```python
from slovakey import SlovaKey
sk = SlovaKey()
keywords = sk.extract_keywords([text])
