# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Untersuchung und Vorbereitung der Daten

# %% [markdown]
# ## Allgemeines

# %% [markdown]
# Die vorliegenden Daten ([https://www.kaggle.com/datasets/timmate/avocado-prices-2020?resource=download](https://www.kaggle.com/datasets/timmate/avocado-prices-2020?resource=download])) enthalten Verkaufspreise für Avocados wöchentlich zusammengetragen für unterschiedliche geographische Angaben (Städte, Bundesstaaten, Regionen, ...) in den USA.
# Angegeben ist der durschnittliche Preis einer Avocado (Spalte `'average_price'`) an einem bestimmten Tag (`'date'`). Weiter ist zum Beispiel genannt, ob es sich um konventionell oder biologisch angebaute Avocados handelt (`'type'`).

# %% [markdown]
# Bei den Spalten `'4046'`, `'4225'`, und `'4770'`, handelt es sich um Price Lookup (PLU) Codes:  
#  - Small/Medium Hass Avocado (~3-5oz avocado) | #4046 Avocado
#  - Large Hass Avocado (~8-10oz avocado) | #4225 Avocado
#  - Extra Large Hass Avocado (~10-15oz avocado) | #4770 Avocado
#
# Diese werden unten umbenannt.

# %% [markdown]
# Die Daten mit Ortsbezeichung `geography="Total U.S."` sind laut Beschreibung aggregiert, ergeben sich aber nicht aus den Daten der Unterregionen.

# %%
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
raw_data = pd.read_csv("../data/avocado-updated-2020.csv")
raw_data.rename(columns={'4046': 'S/M', '4225': 'L', '4770': 'XL'}, inplace=True)
raw_data["date"] = pd.to_datetime(raw_data["date"])

# %%
raw_data.head()

# %%
raw_data.info()

# %% [markdown]
# ## Zielsetzung

# %% [markdown]
# Wir versuchen den Avocodo-Preis zu einem bestimmten Zeitpunkt (nächste Woche) auf Grundlage vergangener Preise (diese Woche, letztes Jahr, ...) vorherzusagen. Wir wollen die Vorhersage für unterschiedliche Regionen getrennt machen. 

# %% [markdown]
# ## Untersuchung der Daten

# %% [markdown]
# Mögliche Werte und Häufigkeit für kategorielle Spalten:

# %%
for column in ["type", "year", "geography"]:
    print(f"{column}:")
    print(*(f"\t{k}: {v}" for k, v in Counter(raw_data[column]).items()), sep="\n")

# %%
region_of_interest = "Total U.S."
regional_data = raw_data[raw_data["geography"] == region_of_interest]
regional_data.describe()

# %%
regional_data.hist()
plt.tight_layout()

# %% [markdown]
# Schauen wir uns einmal nur die `'date'` Spalte für eine einzelne Region an. Sind die Abstände der Datenpunkte konsistent? 

# %%
region_of_interest = "Total U.S."
regional_data = raw_data[raw_data["geography"] == region_of_interest]
date = regional_data["date"].sort_values()

# %% [markdown]
# Jedes Datum kommt genau zweimal vor; je einmal für `type='organic'` und `type='conventional'`

# %%
date.duplicated().sum() / date.count()

# %%
date

# %% [markdown]
# Wir berechnen die zeitlichen Abstände zwischen den Datenpunkten in Tagen und sehen, dass es einen sehr großen Abstand von 36 (statt der erwarteten 7) Tage Anfang 2019 gibt.

# %%
differences = (date - date.shift(2)).apply(lambda x: x.days).to_numpy()

# %%
date.iloc[np.argwhere(differences > 10)[0, 0]]

# %%
plt.plot(date.iloc[::2], differences[::2])
plt.title("Zeitlicher Abstand zum letzten Datenpunkt")
plt.xlabel("Datum")
plt.ylabel("Abstand in Tagen")

# %% [markdown]
# Wir können versuchen die fehlenden Werte durch interpolieren zu ergänzen. Da die Abstände auch an anderen Stellen leicht unregelmäßig sind, bietet es sich an dies für alle Tage im betrachteten Zeitraum zu tun.

# %%
conventional_price = regional_data[regional_data["type"] == "conventional"].set_index('date')['average_price']
full_index = pd.date_range(start=conventional_price.index.min(), end=conventional_price.index.max(), freq='D')

# Reindex the dataframe to the full date range, filling missing entries with 0
conventional_price = conventional_price.reindex(full_index)

# %%
conventional_price

# %%
conventional_price.interpolate(method="time", inplace=True)

# %%
conventional_price.plot()
plt.xlabel("Datum")
plt.ylabel("Preis in $")

# %%
pd.plotting.autocorrelation_plot(conventional_price)

# %% [markdown]
# Damit können wir jetzt die Input-Featuers für unsere Vorhersage folgendermaßen generieren: Um den Avocado-Preis für einen bestimmten Tag $x$ vorherzusagen, wollen wir den Preis an einer Reihe von Tage davor heranziehen, d.h. zum Beispiel an Tag $x - 1$, $x - 7$, ... Wir müssen dabei bedenken, dass durch die Auswahl der betrachteten vergangenen Tage festgelegt wird, wie weit in der Zukunft unsere Vorhersagen liegen können.

# %%
conventional_price.shift([0, 1, 7, 14, 21]).head(30)
