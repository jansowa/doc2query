# Bielik doc2query — pakiet instrukcji dla Codex

Punktem wejścia jest [`AGENTS.md`](AGENTS.md). Pliki w `tasks/` są wykonawczymi specyfikacjami kolejnych etapów.

Ten katalog nie zawiera gotowego kodu treningowego. Zawiera kompletną specyfikację projektu, według której system agentowy Codex ma utworzyć repozytorium, testy, skrypty, konfiguracje i raporty eksperymentalne.

Zalecany sposób użycia:

1. skopiuj cały katalog do katalogu głównego nowego repozytorium;
2. uruchom Codex w tym repozytorium;
3. poleć: „Przeczytaj AGENTS.md i wykonaj task 00. Nie rozpoczynaj tasku 01, dopóki kryteria akceptacji tasku 00 nie przejdą.”;
4. kolejne zadania zlecaj osobno, zachowując wyniki i decyzje w repozytorium.
