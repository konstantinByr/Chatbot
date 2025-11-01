# Mercedes-Benz Car Recommendation Chatbot

A German-language conversational chatbot that recommends the perfect Mercedes-Benz model based on user preferences using deep learning and natural language processing.

## Overview

**Maggus** (Marcus) is an intelligent assistant that engages users in a conversation to understand their car preferences across five key dimensions:
- Engine type (Petrol, Diesel, Hybrid, Electric)
- Vehicle type (SUV or non-SUV)
- Vehicle class (Compact to Luxury)
- Body style (Estate, Sedan, Coupe, Convertible)
- Number of doors (2 or 4)

Based on the responses, it calculates a score and recommends the most suitable Mercedes-Benz model from a database of 30+ vehicle configurations.

## Features

- **Natural Language Understanding**: Uses a PyTorch neural network trained on German language patterns
- **Intent Classification**: 19 different user intent categories with 75% confidence threshold
- **Custom German Stemmer**: Implements the CISTEM algorithm for German text preprocessing
- **Intelligent Scoring System**: Weighted scoring across multiple vehicle attributes
- **Interactive Conversation Flow**: Sequential questioning with validation and error handling

## Project Structure

```
Chatbot/
├── src/                    # Source code
│   ├── chat.py            # Main chatbot interaction script
│   ├── train.py           # Model training script
│   ├── model.py           # Neural network architecture
│   └── chatbot3.py        # Text preprocessing utilities
├── data/                   # Data files
│   ├── intents.json       # Intent patterns and responses
│   └── data.pth           # Trained model weights
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Technologies Used

- **Python 3.x**
- **PyTorch**: Neural network framework
- **NLTK**: Natural language processing toolkit
- **NumPy**: Numerical computations
- **Neptune**: ML experiment tracking (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK German stopwords (first time only):
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### Running the Chatbot

Navigate to the `src` directory and run:
```bash
cd src
python chat.py
```

The chatbot will greet you and start asking questions. Answer in German, and type `ENDE` to exit.

**Example Interaction:**
```
Maggus: Hallo, ich bin Maggus, dein persoenlicher MB-Auswahlassistent!
Maggus: Schreibe 'ENDE', um das Programm zu beenden.
Maggus: Welchen Antrieb willst du haben?
Du: Elektrisch
Maggus: [Response about electric vehicles]
Maggus: Möchtest du einen SUV?
Du: Ja
...
```

### Training the Model

To retrain the model with updated intents or parameters:
```bash
cd src
python train.py
```

**Training Parameters:**
- Epochs: 200
- Batch size: 32
- Learning rate: 0.0001
- Hidden layer size: 128
- Optimizer: Adadelta
- Loss function: CrossEntropyLoss

The training script saves the best model to `data/data.pth` when the loss improves.

## How It Works

### 1. Intent Classification

The chatbot uses a feed-forward neural network with the following architecture:
```
Input Layer → Hidden Layer (128) → Hidden Layer (128) → Output Layer
```

User input is:
1. Tokenized into individual words
2. Stemmed using the custom German CISTEM algorithm
3. Converted to a bag-of-words representation
4. Passed through the neural network
5. Classified into one of 19 intent categories

### 2. Scoring System

Each preference contributes to a final score:
- **Antrieb (Engine)**: 10,000 - 40,000 points
- **SUV**: 0 - 1,000 points
- **Klasse (Class)**: 0 - 400 points
- **Karosserie (Body)**: 0 - 30 points
- **Türen (Doors)**: 0 - 1 points

**Formula**: `Score = Engine + SUV + Class + Body + Doors`

### 3. Car Recommendation

The final score maps to specific Mercedes-Benz models:
- A-Class, B-Class (Compact)
- C-Class, E-Class, S-Class (Sedans)
- GLA, GLB, GLC, GLE, GLS (SUVs)
- CLE (Coupe/Cabriolet)
- AMG GT, SL (Sports cars)
- Maybach (Ultra-luxury)
- EQ models (Electric variants)

## Intent Categories

The chatbot recognizes 19 different intents:

| Category | Intents | Examples |
|----------|---------|----------|
| Greetings | Begruessung, Verabschiedung | "Hallo", "Tschau" |
| Engine | Benzin, Diesel, Hybrid, Elektrisch | "Benziner", "E-Auto" |
| Type | SUV, kein SUV | "Geländewagen", "Nein" |
| Class | Kompakt, Mittel, obere Mittel, Oberklasse, Luxusklasse | "Mittelklasse", "Maybach" |
| Body | T-Modell, Limousine, Coupe, Cabrio | "Kombi", "Cabriolet" |
| Doors | 2Tuerer, 4Tuerer | "Zweitürer", "4 Türen" |

## Model Architecture

The neural network is defined in `src/model.py`:

```python
class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, outputSize)
        self.relu = nn.ReLU()
```

## Text Preprocessing

The `chatbot3.py` module provides:

- **tokenize()**: Splits German text into words using NLTK
- **stem()**: Custom German stemmer using CISTEM algorithm
  - Handles German special characters (ä, ö, ü, ß)
  - Implements language-specific stemming rules
- **bagOfWords()**: Converts tokenized text to binary feature vectors

## Customization

### Adding New Intents

Edit `data/intents.json`:

```json
{
  "tag": "NewIntent",
  "patterns": [
    "pattern 1",
    "pattern 2"
  ],
  "responses": [
    "response 1",
    "response 2"
  ]
}
```

Then retrain the model:
```bash
cd src
python train.py
```

### Adding New Cars

Modify the `autos` dictionary in `src/chat.py`:

```python
autos = {
    score: 'Model Name',
    # ... more mappings
}
```

### Adjusting Confidence Threshold

In `src/chat.py`, modify:
```python
wahrscheinlichkeit = 0.75  # 75% confidence threshold
```

## Development

### Experiment Tracking

The project includes Neptune AI integration (currently disabled). To enable:

1. Uncomment Neptune code in `src/train.py`
2. Add your API token
3. Configure project settings

### Model Checkpointing

During training, the model automatically saves when loss improves:
- Best model: `data/data.pth`
- Intermediate saves: `data/data3.pth`

## Known Limitations

- Only supports German language input
- Requires exact intent matches (>75% confidence)
- Limited to predefined Mercedes-Benz model lineup
- No support for additional customization preferences (color, transmission, etc.)

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (especially in German)
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- German stemmer based on CISTEM algorithm by Leonie Weissweiler
- Mercedes-Benz model data (educational use only)
- Built with PyTorch and NLTK

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Bot Name**: Maggus (Marcus)
**Language**: German
**Framework**: PyTorch
**Last Updated**: 2025
