## Temporally-Extended Epsilon-Greedy Exploration

*Will Dabney, Georg Ostrovski & André Barreto (DeepMind), ICLR 2021*

---

### Abstract

Das Paper beschreibt ein Problem des klassischen epsilon-greedy Verfahrens, das dazu neigt, die Erkundung nicht dauerhaft beizubehalten (sogenanntes Dithering). Die Autorinnen und Autoren schlagen vor, bei der Erkundung eine zufällige Aktion über mehrere Schritte hinweg beizubehalten. Diese einfache Änderung verbessert die Exploration in vielen Umgebungen, ohne die Einfachheit oder Konvergenzeigenschaften von epsilon-greedy zu beeinträchtigen.

---

### Kernidee

* **Standard epsilon-greedy:**

  * Mit Wahrscheinlichkeit (1 minus epsilon) wird die Aktion mit dem höchsten Schätzwert gewählt, andernfalls wird zufällig eine einzelne Aktion ausgeführt.
* **Temporally-Extended Epsilon-Greedy:**

  * Es wird eine Menge von Optionen definiert, wobei jede Option bewirkt, dass eine bestimmte Aktion für eine festgelegte Anzahl von Schritten wiederholt wird.
  * Mit Wahrscheinlichkeit (1 minus epsilon) wird klassisch greedy gehandelt, mit Wahrscheinlichkeit epsilon wählt man eine dieser Optionen und führt sie bis zu ihrem Ende aus.
  * Wenn die Anzahl der Wiederholungsschritte gleich eins ist, entspricht das genau dem Standardverfahren.

---

### Dauerverteilungen für Wiederholungsschritte

Die Dauer, wie lange eine Aktion wiederholt wird, kann aus verschiedenen Wahrscheinlichkeitsverteilungen gezogen werden:

* Eine gleichverteilte Auswahl zwischen 1 und einer Maximalzahl N.
* Eine einfache exponentielle Verteilung, die tendenziell viele kurze Wiederholungen erzeugt.
* Eine Verteilung mit schwerem Schwanz, die gelegentlich sehr lange Wiederholungen liefert.

In Experimenten zeigte sich, dass die schwer-schwänzige Verteilung oft den besten Kompromiss zwischen kurzen und langen Aktionssequenzen bietet.

---

### Theoretische Eigenschaften

* **Besuchszeit aller Zustand-Aktion-Kombinationen:**
  Temporally-extended Optionen können die Zeit, bis jeder Zustand mit jeder Aktion besucht wurde, unter bestimmten Annahmen polynomiell beschränken.
* **Konvergenz:**
  Die veränderte Erkundungsstrategie behält die theoretischen Garantien von Q-Learning mit epsilon-greedy bei, da die Optionen lediglich den Zustand um vergangene Aktionen erweitern.

---

### Experimentelle Ergebnisse

* **Atari-57 Benchmark (Deep RL):**

  * Eingebettet in die Agenten Rainbow und R2D2 wurden die modifizierten Explorationsregeln gegen Standard epsilon-greedy, Pseudo-Counts, RND und Bootstrapped DQN verglichen.
  * Bei technisch anspruchsvollen Spielen wie Montezuma’s Revenge oder Pitfall erzielten die temporally-extended Regeln deutlich höhere Punktzahlen, ohne dass die Leistung bei leichteren Spielen litt.

---

### Fazit

Temporally-Extended Epsilon-Greedy ist eine einfache, aber effektive Erweiterung von Standard epsilon-greedy. Sie verbessert die Ausdauer der Erkundung und skaliert von kleinen Markov-Modellen bis zu komplexen Deep-RL-Szenarien, ohne aufwändige Anpassungen an bestehenden Algorithmen.

---  English ------------------------------------------------------------------------------------------------------------------------

## Temporally-Extended Epsilon-Greedy Exploration

*Will Dabney, Georg Ostrovski & André Barreto (DeepMind), ICLR 2021*

---

### Abstract

The paper identifies a limitation of the classic epsilon-greedy method, which tends to abandon exploration too quickly, causing so-called dithering. The authors propose to maintain an exploratory action for multiple consecutive steps. This simple modification improves exploration in various environments without compromising the simplicity or convergence guarantees of epsilon-greedy.

---

### Core Idea

* **Standard epsilon-greedy:**

  * With probability (1 minus epsilon), select the action with the highest estimated value; otherwise, choose a random single action.
* **Temporally-Extended epsilon-greedy:**

  * Define a set of options, each specifying that a chosen action is repeated for a fixed number of steps.
  * With probability (1 minus epsilon), act greedily; with probability epsilon, sample one of these options and follow it to completion.
  * If the repeat count equals one, this reduces to the standard epsilon-greedy algorithm.

---

### Repeat Duration Distributions

The number of consecutive repetitions for an exploratory action can be sampled from various distributions:

* A uniform distribution between 1 and a maximum value N.
* An exponential distribution, favoring many short repeats.
* A heavy-tailed distribution, occasionally producing very long repeats.

Experiments show that a heavy-tailed distribution often offers the best trade-off between short and long action sequences.

---

### Theoretical Properties

* **Cover Time for State-Action Pairs:**
  Temporally-extended options can ensure that every state-action pair is visited within polynomial time under certain assumptions.
* **Convergence:**
  The modified exploration strategy preserves the convergence guarantees of Q-learning with epsilon-greedy, as options merely augment the state with recent action history.

---

### Experimental Results

* **Atari-57 Benchmark (Deep RL):**

  * Integrated into Rainbow and R2D2 agents, the modified exploration rules were compared against standard epsilon-greedy, pseudo-counts, RND, and bootstrapped DQN.
  * On challenging games like Montezuma’s Revenge and Pitfall, temporally-extended exploration achieved significantly higher scores without degrading performance on easier games.

---

### Conclusion

Temporally-Extended epsilon-greedy is a simple yet effective enhancement of standard epsilon-greedy. It increases exploration persistence and scales from small tabular MDPs to complex deep RL settings without requiring substantial changes to existing algorithms.
