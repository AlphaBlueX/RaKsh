# Borewell Feasibility Checker

## Overview

This project is an ML-powered web application that helps determine whether a borewell can be safely and sustainably constructed at a specific geographic location. Users interact with a map-based frontend, click on a desired location, and receive instant feedback from a machine learning model trained on environmental and geospatial data.

## Features

- *Interactive Map*: Click to select any location and get prediction.
- *ML Model*: Random Forest classifier trained on features like:
  - Rainfall history
  - Soil type
  - Groundwater level
  - Existing borewells
  - Population density
- *Water Usage Section*: View simulated usage data per borewell.
- *Responsive UI*: Real-time feedback with minimal delay.

## Tech Stack

- *Frontend*: HTML, CSS, JavaScript, LeafletJS
- *Backend*: Python, Flask
- *ML Model*: Scikit-learn (Random Forest)
- *Data*: Sample dataset with 100+ rows from Bengaluru (editable for other regions)

## Folder Structure
