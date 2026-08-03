# Building Classification Using a YOLO Object Detection Model Trained on Aerial Imagery

**Research Lab:** [BYU Transportation Lab](https://github.com/byu-transpolab)  

## Project Overview
This project researches whether and to what extent remote sensing data can be used to develop land use and socioeconomic data independent of traditional statistical agencies. The development of a method to synthesize such data would solve significant problems in regions without reliable statistical infrastructure and provide a high-frequency supplement to data in established areas.

Currently, the project utilizes a **YOLO (You Only Look Once)** object detection framework trained to identify and categorize building footprints within urban environments.

This code was used to write the associated paper, and was used as launching off point for this project [point_cloud_maker](https://github.com/cojow/point_cloud_maker).

---

## Future Research Directions

To increase the accuracy of building categorization and transition toward predictive socioeconomic modeling, we are exploring several key areas:

### 1. High-Fidelity Data & Drone Integration
While datasets like Microsoft Building Footprints are expansive, they are often several years old and lack recent developments. We are moving toward:
* **Drone-Acquired Imagery:** Using commercially available drones to capture higher-quality, fine-grained imagery that allows for house distinction at a much more detailed level.
* **Multi-Angle Perspectives:** Leveraging the ability of drones to capture images from multiple angles to create a more complete picture of an area and extract more accurate footprints than those available in existing public datasets.

### 2. Multi-Variable Predictive Modeling
This involves a general model that predicts socioeconomic data (e.g., household income, occupancy) by incorporating the YOLO output as one of many inputs:
* **Socioeconomic Proxies:** Training specialized models to identify objects like vehicles, swing sets, and foliage density.
* **Multi-Spectral Data:** Integrating thermal radiation (heat use), nighttime lights, and vegetation indices (NDVI) to infer household-level statistics.
* **Spatial Contextualization:** Analyzing neighboring building classifications to improve accuracy—such as distinguishing a residential structure from a commercial support building based on its surrounding environment.

### 3. Logistical & Ethical Considerations
As we move toward localized data collection, our research also addresses the limitations inherent in remote sensing:
* **Data Consistency:** Mitigating the issues of outdated or inconsistent satellite imagery in fast-developing areas.
* **Logistics & Weather:** Planning for flight logistics and weather dependency for drone operations.
* **Privacy & Ethics:** Navigating the ethical line between public and private information, ensuring that drone-collected data adheres to local and national regulations.

---

## Repository Structure
* `/data`: Sample building footprints and imagery subsets.
* `/models`: YOLO configuration and weights for building classification.
* `/scripts`: Python scripts for data processing and model training.

## Getting Started
1. Clone the repository: 
   `git clone https://github.com/byu-transpolab/dronemodeling_Logan.git`
2. Install the necessary Python environment:
   `pip install -r requirements.txt`
