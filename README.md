# Town and Data Centre Water Model
 
A Streamlit app that screens what a proposed data centre does to a town's water supply in a dry year, and what could be done about it.
 
It couples a town and a data centre in one daily reservoir model, runs Monte Carlo simulations, compares source options, and turns the results into a verdict, approval conditions and a plain-language public report.
 
**Live app:** add your Streamlit link here
 
> This is a screening tool with simplified assumptions. It is not a forecast and it is not calibrated to any real place. Treat the results as a way to compare options and ask better questions, not as engineering design.
 
## What it does
 
- Models town demand that changes with season, weekday, heat and population growth
- Simulates a reservoir on a daily timestep over 5 to 20 years, including a drought period
- Adds a data centre whose cooling water depends on cooling mode, cycles of concentration and air temperature
- Couples the two in both directions: the data centre draws down the reservoir, and the reservoir level decides the data centre's cooling mode and source mix
- Reports power-linked (indirect) water separately from site water
- Compares four source options and recommends the best one
### Source options
 
| Option | Description |
|---|---|
| Baseline | Local stream only |
| A | Reclaimed (treated wastewater) for cooling |
| B | Added groundwater source, used when the reservoir is in Amber |
| C | Combined package: reclaimed water, groundwater, leak reduction, higher cycles of concentration, adaptive cooling, and curtailing the data centre before households |
 
## App tabs
 
- **Overview:** verdict (Green, Amber, Red), four headline KPIs, a short generated analysis, a monthly animation of the model run, and delivered-demand chart
- **Compare options:** scorecard for all options, reservoir levels, and an uncertainty band across Monte Carlo runs
- **What must be done:** approval conditions, technology ranking by cost per net litre saved, corrective action report, escalation ladder and water-quality flags
- **Community desk:** report form with response deadlines, complaint-cluster check, early-warning sample data, and an audit check of reported against modelled draw
- **Public report:** stress level, cooling mode and source mix for a typical or worst month, with downloadable text and CSV
### KPIs
 
1. Shortfall probability: share of simulated years with at least one day when demand cannot be met
2. Restriction days per year
3. Data centre share of drought-year supply, against a policy cap
4. Cost per kL secured: extra yearly cost divided by kL of stream water avoided versus baseline
## Run locally
 
Requires Python 3.10 or newer.
 
```bash
python -m venv .venv
 
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS or Linux
source .venv/bin/activate
 
pip install -r requirements.txt
streamlit run app.py
```
 
The app opens at http://localhost:8501.
 
## Project structure
 
```
app.py              Streamlit interface: inputs, charts, tabs
engine.py           All calculations, no UI code (numpy and pandas only)
requirements.txt    Python dependencies
.streamlit/
  config.toml       Dark theme
```
 
`engine.py` can be used on its own from a notebook or script:
 
```python
import engine as E
 
P = dict(E.DEFAULTS)
P["it_mw"] = 150
results = E.run_all(P)
 
print(results["best"])
print(results["C"]["kpi"])
```
 
## How the model works
 
Daily water balance for the town reservoir:
 
```
S(t+1) = S(t) + inflow - environmental flow - town draw - data centre draw - losses
```
 
- **Demand** is not constant. It follows a seasonal cycle, a weekly cycle, a temperature response and population growth.
- **Data centre water** = IT load x water use per kWh x cooling-mode factor x cycles-of-concentration factor x temperature factor. Cycles of concentration are capped by feed-water hardness.
- **Cooling mode** switches between evaporative, hybrid and dry according to the reservoir level (Amber and Red thresholds). If air temperature is too high for dry cooling, the model falls back to hybrid cooling plus workload shifting.
- **Source selector** draws reclaimed water first, then groundwater when the reservoir is in stress, then the stream.
- **Reclaimed water** depends on the town's treated wastewater, which depends on the water delivered.
- **Indirect water** = electricity use x grid water intensity, tracked separately and not added to the town's balance.
- **Monte Carlo:** 50 to 500 runs vary inflow, temperature and demand. Run 0 is a clean central case used for the animation.
- **Verdict:** Red if shortfall probability exceeds 25% or restriction days exceed 45 per year. Amber above 5% or 7 days. Green otherwise.
## Inputs
 
All inputs are in the sidebar: town size and leakage, reservoir size, stream yield, climate, drought timing and severity, operating rules, data centre size and cooling, and the capacity of each source option. Three presets are included as starting points. Change any value, then press **Run simulation**.
 
## Assumptions and limits
 
- Several parameters are assumptions, not data: cooling-mode multipliers, cost per ML of each source, lever unit costs and energy use. They are defined at the top of `engine.py` in `ASSUMPTIONS`, `COSTS`, `FIXED` and `LEVERS`. Replace them with local values.
- The grid water intensity default (4.52 L/kWh) is a US average. Set your own grid's value in the sidebar.
- The town is a single lumped reservoir. There is no spatial groundwater model and no hourly operation. Peak-hour effects are not captured.
- Community reports and the audit check run in the browser session only and are not stored.
- Sample early-warning reports are generated by the model for demonstration.
## Deploy
 
The app deploys on Streamlit Community Cloud:
 
1. Push this repository to GitHub (public)
2. Go to share.streamlit.io and sign in with GitHub
3. Choose **Create app**, select this repository, branch `main`, main file `app.py`
4. Click **Deploy**
GitHub Pages cannot host Streamlit apps because it only serves static files.
 
## Customising
 
- Change defaults and presets in `DEFAULTS` and `PRESETS` in `engine.py`
- Change thresholds for the verdict in `verdict()`
- Add a technology lever by adding an entry to `LEVERS` and a call to `add(...)` in `recommend()`
- Change the theme in `.streamlit/config.toml`
