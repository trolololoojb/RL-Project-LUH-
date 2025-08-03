# Insurance Gym – Kurzbeschreibung

## 1. Zweck der Umgebung

Die Datei `insurance_gym.py` simuliert eine **vereinfachte Underwriting-Umgebung** für Versicherungskunden. Bei jedem Schritt erhält der Agent ein neues Kundenprofil und muss entscheiden:

* **Aktion 0 – ablehnen** → keine Transaktion (kein Geld, kein Risiko)
* **Aktion 1 – annehmen** → sofortige Gutschrift der Prämie, später evtl. Schadenszahlung

Ziel des Agenten: **möglichst viel nominalen Gewinn** (Prämien minus Schäden) über eine Episode sammeln. Die zeitliche Abzinsung übernimmt der RL-Agent selbst.

---

## 2. Kunden-Profil

Jeder Kunde wird als Instanz der Klasse `Profile` erzeugt mit drei Merkmalen:

| Merkmal      | Wertebereich            | Beschreibung                                     |
| ------------ | ----------------------- | ------------------------------------------------ |
| `age`        | ganzzahlig in \[18, 80] | Alter des Kunden                                 |
| `region`     | ganzzahlig in \[0, 4]   | Region (0: sehr niedriges Risiko … 4: sehr hoch) |
| `risk_score` | gleitkomma in \[0, 1)   | Zufälliger Risikowert                            |

Die Profile werden bei Env-Initialisierung nach einem fixen `seed` erstellt, um Reproduzierbarkeit zu gewährleisten.

---

## 3. Prämienberechnung

Die jährliche Prämie für ein Profil `p` wird deterministisch berechnet:

```
premium(p) = 200 + 4 * (p.age - 18)
             + [0, 20, 40, 60, 80][p.region]
```

Je älter der Kunde und je riskanter die Region, desto höher die Prämie.

---

## 4. Schaden-Wahrscheinlichkeit

Die Wahrscheinlichkeit, dass ein Schaden auftritt, ergibt sich aus:

```
prob(p) = 0.02
          + 0.25 * p.risk_score
          + (p.age - 18) / (80 - 18) * 0.10
          + [0, 0.01, 0.03, 0.05, 0.07][p.region]
```

Um Ausreißer zu vermeiden, wird der Wert auf max. 0.90 gekappt:

```
claim_prob(p) = min(prob(p), 0.90)
```

---

## 5. Zustandsraum: Binning

Zur Reduktion der Zustände werden die Features diskretisiert:

* `N_AGE_BINS = 13`
* `N_REGIONS = 5`
* `N_RISK_BINS = 10`

Die Gesamtzahl der Zustände beträgt `13 × 5 × 10 = 650`. Das Environment stellt dies als `gym.spaces.Discrete(650)` zur Verfügung.

---

## 6. Aktionen und Reward

* **Aktionen**: `gym.spaces.Discrete(2)` (0=ablehnen, 1=annehmen)
* **Reward**:

  * Bei Annahme: sofortige Auszahlung der Prämie
  * Bei Ablehnung: 0
  * Schadenszahlungen werden nach einem festen `delay` (Standardwert 10) verzögert ausgezahlt.

Intern verwendet das Environment eine `collections.deque` der Länge `delay + 1`, um Rewards verzögert zu liefern. Die maximale Episodenlänge (`horizon`) beträgt 1 000 Schritte.

---

## 7. Pareto-Schadensverteilung & Parameter

| Parameter      | Default-Wert | Beschreibung                          |
| -------------- | ------------ | ------------------------------------- |
| `n_profiles`   | 100          | Anzahl gleichzeitiger Profile         |
| `delay`        | 10           | Verzögerung der Schadensauszahlung    |
| `horizon`      | 1000         | Maximale Episodenlänge                |
| `pareto_alpha` | 1.5          | Formparameter der Pareto-Verteilung   |
| `pareto_xm`    | 1.0          | Skalenparameter der Pareto-Verteilung |
| `seed`         | –            | Zufalls-Seed für Profile und Claims   |

Schadensbeträge werden bei Auftreten aus einer Pareto-Verteilung (`alpha=1.5`, `xm=1.0`) gezogen.

---

## 8. Warum ist das lernbar?

* **Begrenzter Zustandsraum (650 Zustände)** → überschaubare Q-Tabelle
* **Deterministische Regeln** in Prämie & Schaden → klare Strukturen, die Agent erlernen kann
* **Verzögerte Belohnung** durch `delay` → lohnende Exploration trotz kurzfristiger Ungewissheit

---

### TL;DR

`insurance_gym.py` bietet eine **kompakte**, aber **realistische** Simulationsumgebung, in der ein RL-Agent Underwriting-Entscheidungen trifft. Die Kombination aus deterministischen Prämienregeln, stochastischer Schadenserzeugung, verzögerter Auszahlung und fetten Pareto-Ausreißern macht das Training herausfordernd und spannend.
