# RL-Project-LUH-

In a synthetic insurance-underwriting MDP with Pareto heavy-tailed losses and delayed payouts, does rate-corrected, k-persistent ε-greedy (EZ) achieve higher discounted return and lower ruin risk (bankruptcy rate / CVaR of episodic returns) than fixed and annealed ε-greedy, under the same DQN and training budget?


# Setup
1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ````
2. **Activate the virtual environment:**
    - On Windows:
    ```bash
    .venv\Scripts\activate
    ```
    - On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```

3. **Install dependencies from ```requirements.txt```:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the experiment script:**
    ```bash
    python run_experiment.py [OPTIONS]
    ```

    without specified options, uses every default parameter, like in our experiment.

    **Available parameters**:

    ```--episodes```, ```-p``` (int, default=500): Number of episodes per seed

    ```--horizon```, ```-l``` (int, default=500): Maximum number of steps per episode

    ```--delay_min```, ```-dmi``` (int, default=8): Min Delay parameter for claims

    ```--delay_max```, ```-dma``` (int, default=25): Max Delay parameter for claims

    ```--k_repeat```, ```-k``` (int, default=4): Repetition length for EZ-Greedy (defaults to delay)

    ```--eps```, ```-e``` (float, default=0.05): Base ε for fixed ε-greedy

    ```--alpha```, ```-a``` (float, default=0.0004): Learning rate α

    ```--gamma```, ```-g``` (float, default=0.999): Discount factor γ

    ```--seed```, ```-s``` (int, default=10): Starting seed (uses seed, seed+1, … seed+9)

    ```--scale```, ```-c``` (int, default=1): EZ Scaling (0 for no scaling, 1 for scaling)

    ```--bankruptcy_penalty```, ```-bp``` (float, default=300000.0): Extra penalty applied on bankruptcy

    ```--hidden_dims```, ```-hd``` (str, default="256,256"): Comma-separated hidden layer sizes for the DQN, e.g. "128,128" or "256,256,128"

    ```--batch_size```, ```-bs``` (int, default=128): Batch size for DQN updates

    ```--sync_every```, ```-se``` (int, default=750): Target network sync interval in environment steps

    ```--buffer_capacity```, ```-bc``` (int, default=200000): Replay buffer capacity (number of transitions)

    ```--eps_start```, ```-es``` (float, default=1.0): Starting ε for the annealed schedule

    ```--eps_end```, ```-ee``` (float, default=0.05): Final ε for the annealed schedule

    ```--eval_every``` (int, default=50): Run an evaluation block every N training episodes

    ```--eval_episodes``` (int, default=50): Number of eval episodes per evaluation block


5. **Run analysis notebook:**

    Open and run the Jupyter notebook:
    ```bash
    jupyter notebook insurance_experiment_analysis.ipynb
    ```

# More insights
[Plots](trolololoo.de)


# Johann, der Agentenbändiger

Inmitten von Code, so still und klar,
sitzt Johann, der stets Forscher war.
Mit States und Actions, Reward im Blick,
trainiert er den Agenten Stück für Stück.

Er schaut, wie Policy langsam lernt,
wie Q-Werte wachsen, wie alles sich entfernt
vom Zufall, vom reinen, blinden Tun –
bis Strategien wie Sterne am Himmel ruh’n.

Manch Episode schlägt fehl, doch er lacht,
denn Fehler sind’s, die Fortschritt machen sacht.
Geduldig passt er Hyperparameter an,
bis der Agent sein Ziel erreichen kann.

So sitzt er bis spät, bei Kerzenschein,
in einer Welt aus Zuständen, groß und klein,
und denkt sich: „Wie schön, wenn Maschinen verstehen –
doch am schönsten ist’s, den Weg dorthin zu gehen.