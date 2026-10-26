# Server locale S1 + S2 sul desktop

Due server separati sulla stessa GPU da 12 GB, nessun gateway:

| | Cosa | Porta | Formato |
|---|---|---|---|
| **S2** narratore | LM Studio (motore llama.cpp) con Qwen3-4B-Instruct-2507 Q4_K_M | 1234 | OpenAI chat completions |
| **S1** decisore | Kev-4B (`kev.serve`), stesso formato di Jev | 8009 | `POST /v1/systemone` |

`bench.py` controlla i server e confronta più modelli con dati mock (89 decreti etichettati da un Jev reale per S1, 12 semi di eventi per S2). Usa solo la libreria standard di Python.

## 1. Driver NVIDIA (una volta)
```bash
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
                 https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
# riavvia e controlla (con Secure Boot attivo serve prima registrare la chiave del modulo)
nvidia-smi
```

## 2. S2 in LM Studio
1. Scarica **Qwen3-4B-Instruct-2507** in GGUF **Q4_K_M** (circa 2,5 GB). Non ha la modalità thinking.
2. Caricalo con opzioni manuali: GPU offload al massimo, **Context Length 16384**, **Max Concurrent Predictions 8**, Flash Attention attivo. Se c'è l'opzione, K/V cache in `q8_0`.
3. Developer, Start Server (porta 1234). Spegni il caricamento just-in-time dei modelli, così resta caricato.
4. Senza interfaccia, il demone si usa con `lms daemon up`, `lms server start`, `lms get <modello>` (`lms load --help` mostra le opzioni di caricamento).
5. Prova: `curl -s localhost:1234/v1/models`. Il valore di `id` va in `bench.toml` come `model`.

## 3. S1 con Kev
```bash
./setup-desktop.sh                     # clona Kev, uv sync, crea e avvia il servizio utente tle-s1 su 127.0.0.1:8009
journalctl --user -u tle-s1 -f         # il primo avvio scarica il checkpoint
```
Manuale, senza servizio: `cd ~/tle-local/kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`.
Cambiare checkpoint: `./setup-desktop.sh <altro> 8009` (id Hugging Face o cartella locale; in `kev.serve` il parametro è `--run`).

**La memoria è il punto critico.** Kev-4B in bf16 pesa circa 8 GB e il suo server non documenta opzioni di quantizzazione. Con S2 (circa 4 GB con la cache) il totale supera i 12 GB. Procedi così:
1. Avvia S1 da solo e guarda `nvidia-smi`. Poi carica S2 e riguarda. Se entrambi stanno, hai finito.
2. Se manca memoria: prova un checkpoint più piccolo (Kev-0.8B) oppure ripiega S2 a contesto 8192 con 4 richieste.
3. Se vuoi Kev-4B a 8 bit serve modificare il caricamento dei pesi in Kev (bitsandbytes). Non è provato: usa `bench.py` per confrontare la qualità prima e dopo.

## 4. Test e benchmark
```bash
# modifica le voci [[s1]] e [[s2]] di bench.toml (oppure copia il file e usa --config)
python3 bench.py check                  # una richiesta a ogni server, mostra anche la VRAM
python3 bench.py s1                     # accuratezza su pilastro, dominio, aggressività + latenza e richieste al secondo
python3 bench.py s2                     # JSON valido, assenza di numeri, latenza, token al secondo
python3 bench.py all --levels 1,8,32    # entrambi, con più concorrenza (un turno reale ha fino a 40 richieste S1)
python3 bench.py selftest               # prova lo script con finti server, senza GPU
```
Per confrontare più modelli: aggiungi una voce `[[s1]]` o `[[s2]]` per ciascuno, carica il modello, esegui con `--only <nome>`, cambia modello e ripeti. I risultati finiscono in `results/` (Markdown e JSON). La VRAM misurata è quella dell'intera GPU, quindi include l'altro server.

Come leggere i numeri:
- **S1:** la verità è il dato etichettato da Jev, non il vero valore. Accuratezza sul pilastro sotto circa 0,8 vuol dire che il gioco ricadrà spesso sulla regola simulata. "batched" è l'accuratezza con 5 decreti per richiesta.
- **S2:** "valid and clean" è la quota di risposte con il JSON del contratto, senza cifre (S2 non produce numeri) e di lunghezza limitata. Sotto circa 0,98 il gioco ripete le richieste.

## 5. Collegare il Beelink
Kev ascolta solo su 127.0.0.1 e non ha autenticazione, quindi la via più semplice è un tunnel SSH dal Beelink (funziona per entrambi i server):
```bash
ssh -N -L 8009:127.0.0.1:8009 -L 1234:127.0.0.1:1234 DESKTOP
```
Poi in `config/engines.toml` del progetto:
```toml
[s1]
kind = "jev"
base_url = "http://127.0.0.1:8009/v1/systemone"
model = "kev-latest"

[s2]
kind = "openai_compat"
base_url = "http://127.0.0.1:1234/v1"
model = "qwen3-4b-instruct-2507"
```
Se preferisci raggiungerli direttamente in rete locale, abilita il server di rete in LM Studio e apri solo la tua sottorete: `sudo firewall-cmd --add-rich-rule='rule family=ipv4 source address=192.168.1.0/24 port port=1234 protocol=tcp accept'`. Per Kev non so se esiste un'opzione per l'indirizzo di ascolto: resta sul tunnel.

Controllo dal Beelink: `uv run tle run --engines config/engines.toml --strategy gradual --n 60 --cache runs/cache.jsonl`, poi `uv run tle runs show <run_id> --table execution` (chiamate, tentativi e ripieghi per task).
