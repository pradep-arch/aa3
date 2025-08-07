# SteamNoodles Feedback Agent

## Project Overview

This project implements a multi-agent framework for the **SteamNoodles** restaurant chain to automate customer feedback processing and sentiment visualization. It includes two agents:

1. **Customer Feedback Response Agent**: Analyzes the sentiment of customer reviews (positive, negative, neutral) using the Groq API and LlamaIndex, and generates polite, context-aware responses.
2. **Sentiment Visualization Agent**: Generates bar plots of sentiment trends over a specified date range using pandas and matplotlib.

**Author**: \[Your Name\]\
**University**: \[Your University\]\
**Year**: \[Your Year, e.g., 2025\]

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/[your-username]/steamnoodles-feedback-agent-[your-name].git
   cd steamnoodles-feedback-agent-[your-name]
   ```

2. **Install Dependencies**:

   ```bash
   pip install llama-index llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq pandas matplotlib seaborn
   ```

3. **Set Up Groq API Key**:

   - Obtain a free API key from `https://console.groq.com/keys`.
   - Replace `"YOUR_API_KEY"` in `steam_noodles_agent.py` with your API key.

4. **Prepare Dataset**:

   - Place the `restaurant_reviews.csv` file in the `data/` folder. A sample dataset is provided in the repository (`data/restaurant_reviews.csv`).

5. **Run the Script**:

   ```bash
   python steam_noodles_agent.py
   ```

## Testing the Agents

### 1. Customer Feedback Response Agent

- **Sample Prompt**:

  ```python
  agent.chat("The food was amazing and the service was quick!")
  ```
- **Sample Output**:

  ```
  Sentiment: positive
  Response: Thank you for your kind words! We're thrilled you enjoyed the food and service.
  ```

### 2. Sentiment Visualization Agent

- **Sample Prompt**:

  ```python
  agent.chat("Plot sentiment trends from 2025-08-01 to 2025-08-04")
  ```
- **Sample Output**:

  ```
  Plot saved at: ./data/sentiment_trend_2025-08-01_to_2025-08-04.png
  ```
  - The plot is saved as a PNG file in the `data/` folder. A sample plot is included in the repository (`data/sentiment_trend_2025-08-01_to_2025-08-04.png`).

## Repository Structure

- `steam_noodles_agent.py`: Main script implementing both agents.
- `data/restaurant_reviews.csv`: Sample dataset with reviews, sentiments, and dates.
- `data/sentiment_trend_*.png`: Generated sentiment trend plots.
- `README.md`: Project documentation.

## Dependencies

- Python 3.8+
- llama-index
- llama-index-core
- llama-index-embeddings-huggingface
- llama-index-llms-groq
- pandas
- matplotlib
- seaborn

## Sample Dataset

The sample dataset (`data/restaurant_reviews.csv`) contains the following columns:

- `date`: Date of the review (e.g., 2025-08-01).
- `review`: Text of the customer review.
- `sentiment`: Sentiment label (positive, negative, neutral).

Example:

```csv
date,review,sentiment
2025-08-01,The food was amazing and the service was quick!,positive
2025-08-01,Disappointed with the cold food,negative
2025-08-02,Decent experience, nothing special,neutral
2025-08-02,Best noodles ever!,positive
2025-08-03,Waited too long for my order,negative
2025-08-03,Friendly staff and good ambiance,positive
2025-08-04,Food was okay but overpriced,neutral
```

## Notes

- Ensure the `data/` folder exists and contains `restaurant_reviews.csv`.
- The project uses the Groq free API (`llama-3.1-70b-versatile` model) for LLM tasks.
- The sentiment visualization uses a stacked bar plot for clarity, but can be modified to a line plot if desired.
- For larger datasets, you can use a Kaggle restaurant review dataset (e.g., search "restaurant reviews" on Kaggle).

## 