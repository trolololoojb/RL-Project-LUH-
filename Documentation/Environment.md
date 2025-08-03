# Insurance Gym – Kurzbeschreibung

## 1. Zweck der Umgebung

Die Datei `insurance_gym.py` simuliert eine **vereinfachte Underwriting‑Welt**: Ein Agent (unsere RL‑Strategie) bekommt nacheinander potenzielle Versicherungskunden angeboten und entscheidet bei jedem Kunden:

* **Aktion 0 – ablehnen** → kein Geld, kein Risiko.
* **Aktion 1 – annehmen** → sofort Prämie kassieren, aber später könnte ein Schaden eintreten.

Ziel des Agents: **möglichst viel abgezinsten Gewinn** (Prämien − Schäden) sammeln.

---

## 2. Aufbau eines Kunden („Profile“)

Jeder Kunde ist ein Objekt der Klasse `Profile` und hat genau drei Merkmale:

| Merkmal      | Wertebereich              | Bedeutung                                |
| ------------ | ------------------------- | ---------------------------------------- |
| `age`        | 18 – 80 Jahre (Ganzzahl)  | Alter des Kunden                         |
| `region`     | 0 – 4 (Ganzzahl)          | Fünf grobe Regionen (z. B. Nord, Süd …)  |
| `risk_score` | 0.00 – 1.00 (Dezimalzahl) | Abstrakter Risikowert, höher = riskanter |

> **Anzahl Profile**: Standardmäßig **100**, lassen sich aber über den Parameter `n_profiles` ändern.

### Wie werden Preis & Risiko berechnet?

Die Umgebung benutzt **feste Formeln** (keinen Zufall), sodass dieselben Merkmale immer zu denselben Werten führen – so kann der Agent Regeln *lernen*.

* **Prämie (Einnahme)** wächst mit Alter und Region.
* **Claim‑Wahrscheinlichkeit (Risiko)** steigt mit Risk‑Score, Alter und Region.

---

## 3. Zustand („State“)

Damit tabellarisches Q‑Learning funktioniert, werden die drei Merkmale in **Schubladen (Bins)** einsortiert und dann zu einem einzigen Zustands‑Index (0 – 649) kombiniert:

| Merkmal | Binning                           | Anzahl Bins |
| ------- | --------------------------------- | ----------- |
| Alter   | 5‑Jahres‑Gruppen (18–22, 23–27 …) | 13          |
| Region  | schon diskret 0–4                 | 5           |
| Risk    | 0.0 – 1.0 in 0.1‑Schritten        | 10          |

**Gesamtzustände** = 13 × 5 × 10 = **650**.

---

## 4. Aktionen

```python
self.action_space = spaces.Discrete(2)  # 0 = ablehnen, 1 = annehmen
```

---

## 5. Belohnung (Reward)

1. **Sofort**: Wenn Aktion = 1, wird die berechnete Prämie gutgeschrieben.
2. **Verzögert**: Mit der gespeicherten Claim‑Wahrscheinlichkeit kann nach `delay` Zeitschritten (standard 10) ein **Schaden** eintreten. Die Schadenshöhe stammt aus einer **Pareto‑Verteilung** (sehr hohe Ausreißer möglich).

> Daraus ergibt sich ein typisches Versicherungs‑Problem: hohe, seltene Verluste + zeitliche Verzögerung.

---

## 6. Episode & Parameter

| Parameter                   | Standard       | Bedeutung                                             |
| --------------------------- | -------------- | ----------------------------------------------------- |
| `delay`                     | 10 Schritte    | Wie lange es dauert, bis ein Schaden ausgezahlt wird. |
| `horizon`                   | 1 000 Schritte | Länge einer Episode (max. Kunden pro Durchlauf).      |
| `pareto_alpha`, `pareto_xm` | 1.5, 1.0       | Form der Schadensverteilung (fetter Schwanz).         |
| `seed`                      | –              | Zufalls‑Seed für Reproduzierbarkeit.                  |

---

## 7. Warum ist das lernbar?

* Die Binning‑Strategie beschränkt die **Zahl der Zustände** auf 650 → kleine Tabelle.
* Preis & Risiko hängen **deterministisch** von den Merkmalen ab → es gibt Regeln, die der Agent entdecken kann.
* Durch den **Delay** lohnt sich EZ‑greedy‑Exploration: Eine Aktion bleibt mehrere Schritte aktiv, bis der Schaden sichtbar wird.

---

### TL;DR

`insurance_gym.py` liefert eine **kompakte, aber realistischere** Versicherungs‑Simulation mit 100 synthetischen Kundenprofilen. Zustände werden in Schubladen gepackt, damit tabellarisches Q‑Learning in wenigen Minuten trainiert, während die Aufgabe durch Verzögerung und fette Schadensausreißer spannend bleibt.
