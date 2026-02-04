MatterGen – AI-Assisted Materials Discovery (Prototype)
~ Overview

MatterGen is a prototype web interface built for hackathons to demonstrate the concept of AI-assisted materials discovery. The current version focuses on showcasing the user workflow, data flow, and analytical approach through a lightweight HTML/CSS-based demo.

The full system is planned to be implemented in the final round using Streamlit, machine learning models, and chemistry libraries for real-time analysis and visualization.

~ Problem Statement

Material discovery is slow, expensive, and heavily dependent on trial-and-error experimentation. Researchers lack intuitive tools that allow them to explore material properties, compare candidates, and understand trade-offs at an early stage.

~ Proposed Solution

MatterGen introduces a user-centric platform where researchers can:

Define target material properties

Explore candidate materials and their properties

Understand how close candidates are to the desired requirements

The prototype demonstrates this workflow visually, while the final version will automate it using AI and real datasets.

~ Prototype Features (Current Version)

Clean HTML/CSS-based web interface

Input fields for target material properties

Sample material data for demonstration

Property comparison tables

Visual placeholders for analytics and results

Clear representation of system flow and logic

⚠️ Note: This version is a conceptual and functional demo, not the final AI-powered system.

~ Planned System Architecture

User inputs desired material properties

Backend processes constraints using ML models and materials databases

Candidate materials are generated or fetched

AI-based scoring and ranking is applied

Results are visualized with graphs and molecular structures

~ Tech Stack
Prototype (Current Round)

Frontend: HTML, CSS, JavaScript

Deployment: Web-hosted demo (Replit / static hosting)

Final Implementation (Planned)

Backend & UI: Streamlit

Data & Analysis: Pandas, NumPy

Visualization: Plotly, Py3Dmol

Chemistry Tools: RDKit

Data Sources: Materials Project API

ML Models: Property prediction and similarity scoring

~ Running the Prototype

The prototype is deployed as a static web demo and can be accessed via the provided link.
No local setup is required for the prototype round.

~ Future Scope

Full AI-driven backend with real materials data

Machine learning models for property prediction

2D and 3D molecular/crystal structure visualization

Laboratory-validated datasets

Scalable deployment for research and industry use

⚠️ Limitations

No real ML models or databases are integrated in the prototype

Data shown is illustrative and for demonstration only

Advanced analytics and structure visualization are planned for later stages

~ Team

We are a student team interested in applying AI to scientific and sustainability challenges. Our focus is on building a strong conceptual foundation in the prototype and expanding it into a fully functional system in the final round.

📄 License

This project is developed for educational and hackathon purposes.
