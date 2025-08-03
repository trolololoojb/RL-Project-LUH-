## run\_experiment.py

## 1. Zweck des Skripts

Dieses Skript automatisiert systematische Benchmark-Experimente für zwei Exploration-Strategien in der Underwriting-Umgebung `InsuranceEnv`: Fixed ε-Greedy und EZ-Greedy. Ziel ist es, anhand kumulierter diskontierter Returns und statistischer Kennzahlen (Mittelwert, Standardabweichung) zu untersuchen, wie sich konstante versus adaptive Exploration in einer risikobelasteten Umgebung verhalten.

---

## 2. Module & Abhängigkeiten

* **argparse** – CLI-Parsing für Parameter (Episoden, ε, α, γ, etc.).
* **csv**, **json** – Ein-/Ausgabe von Ergebnis- und Metadatendateien.
* **subprocess** – Ermittlung des aktuellen Git-Commit-Hashes via `git rev-parse HEAD`.
* **pathlib.Path** – Plattformunabhängiges Dateimanagement.
* **statistics** (`mean`, `stdev`) – Aggregation von Returns über verschiedene Seeds.
* **typing** (`Callable`, `List`, `Tuple`) – Typhinweise für `run_single_experiment`.
* **numpy** – Zufallszahlengenerator und numerische Operationen.
* **Projektintern**:

  * `InsuranceEnv` aus `insurance_gym` – Simulationsumgebung für Underwriting mit verzögerten Schadenszahlungen.
  * `fixed_eps_schedule`, `EZGreedy` aus `agents.exploration_schedules` – Implementierungen der beiden Exploration-Mechanismen.
  * `QLearner` aus `agents.qlearner` – Tabelle-basierter Q-Update-Algorithmus.

---

## 3. Grundlegende Logik & Ideen

1. **Exploration vs. Exploitation**

   * Fixed ε-Greedy wählt mit konstanter Wahrscheinlichkeit ε zufällige Aktionen, bleibt aber sonst deterministisch bei der aktuellen Q-Policy.
   * EZ-Greedy passt ε adaptiv an: Anfangs hohe Exploration, später mehr Exploitation basierend auf der Varianz der Returns und dem Verzögerungsparameter. Dadurch soll der Lernprozess in Umgebungen mit verzögerter Belohnung stabiler werden.

2. **Verzögerte Belohnungen**

   * In `InsuranceEnv` werden Prämien sofort gutgeschrieben, Schadenszahlungen jedoch erst nach einer festen Verzögerung (delay) ausgebucht. Dies kann beim Q-Learning zu verzerrten Schätzungen führen.
   * EZ-Greedy berücksichtigt diese Verzögerung als Parameter `k`, um die Exploration gezielt in Phasen mit hohem Informationsgehalt zu steuern.

3. **Reproduzierbarkeit & Robustheit**

   * Mehrfache Durchläufe über verschiedene Seeds (seed … seed+9) reduzieren die Zufallsschwankungen und erlauben statistisch fundierte Vergleiche.
   * Speicherung des Git-Commits garantiert, dass Experimente jederzeit auf denselben Code-Stand bezogen werden können.

4. **Evaluationsmetriken**

   * **Kumulierter diskontierter Return**: Summe der Rewards über alle Episoden, gewichtet mit Discount-Faktor γ.
   * **Mittelwert & Standardabweichung**: Zeigen nicht nur die durchschnittliche Leistung, sondern auch die Stabilität und Varianz der Strategien.

---

## 4. Wichtige Funktionen

### 4.1 get\_git\_commit()

```python
def get_git_commit() -> str:
    # Führt 'git rev-parse HEAD' aus und gibt den Hash zurück oder 'unknown'
```

Dokumentiert den exakten Code-Stand für Reproduzierbarkeit.

### 4.2 run\_single\_experiment(...)

```python
def run_single_experiment(
    schedule_label: str,
    eps_fn: Callable[[int, int], float] | None,
    seed: int,
    episodes: int,
    delay: int,
    gamma: float,
    alpha: float,
    eps: float,
) -> float:
    # 1. Env- und Agent-Initialisierung mit seed
    # 2. Auswahl der Exploration: Fixed (eps_fn) oder EZ (None)
    # 3. Für jede Episode:
    #    a) env.reset(), cumulative_return = 0
    #    b) Solange nicht done:
    #         i.  Aktion wählen
    #         ii. obs, reward, done = env.step(action)
    #         iii. agent.update(obs, action, reward, next_obs)
    #         iv. cumulative_return += reward * (gamma ** timestep)
    # 4. Rückgabe cumulative_return
```

Parameter und Rückgabe wurden im Abschnitt Zweck erläutert.

---

## 5. Hauptprogramm (`main()`)

1. **Argument-Parsing**

   | Option     | Kurzform | Typ   | Default | Beschreibung                         |
   | ---------- | -------- | ----- | ------- | ------------------------------------ |
   | --episodes | -e       | int   | 300     | Anzahl Episoden pro Seed             |
   | --delay    | -d       | int   | 10      | Verzögerung für Schadenszahlungen    |
   | --eps      | -ε       | float | 0.1     | Basis-ε für Fixed ε-Greedy           |
   | --alpha    | -a       | float | 0.1     | Lernrate α                           |
   | --gamma    | -g       | float | 0.99    | Discount-Faktor γ                    |
   | --seed     | -s       | int   | 0       | Start-Seed (es folgen seed … seed+9) |

2. **Verzeichnis & Metadaten**

   * Erstelle `results/` falls nötig.
   * Schreibe `run_meta.json` mit Git-Hash, Parametern und Seed-Liste.

3. **Experiment-Schleife**

   * Strategien: **Fixed** vs. **EZ**
   * Für jeden Seed und jede Strategie `run_single_experiment` ausführen.

4. **Aggregation & Ausgabe**

   * Berechne Mittelwert und Standardabweichung der Returns.
   * Konsolenausgabe im Format:

     ```text
     Strategy: Fixed, Mean Return: 123.45 ± 10.67
     Strategy: EZ,    Mean Return: 130.12 ± 9.23
     ```

5. **Rohdaten-Speicherung**

   * CSV `experiment_summary.csv` mit Spalten: `variant,seed,total_return`.

---

## 6. Datenformate

### 6.1 run\_meta.json

```json
{
  "git_commit": "string",
  "episodes": int,
  "delay": int,
  "eps": float,
  "alpha": float,
  "gamma": float,
  "seeds": [int,...]
}
```

### 6.2 experiment\_summary.csv

| variant | seed | total\_return |
| ------- | ---- | ------------- |
| Fixed   | 0    | 120.5         |
| EZ      | 0    | 130.2         |
| ...     | ...  | ...           |

---

## 7. Beispielaufruf

```bash
python run_experiment.py \
  --episodes 500 \
  --delay 20 \
  --eps 0.05 \
  --alpha 0.1 \
  --gamma 0.99 \
  --seed 42
```

---

## TL;DR

`run_experiment.py` vergleicht Fixed vs. EZ Greedy auf `InsuranceEnv`, erklärt die Exploration-Logik, berücksichtigt verzögerte Belohnungen, sichert Reproduzierbarkeit und liefert statistische Kennzahlen zur Performance-Beurteilung.
